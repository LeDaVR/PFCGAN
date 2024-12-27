import logging

from neuralgym.models import Model # pip install git+https://github.com/JiahuiYu/neuralgym
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.summary_ops import gradients_summary, filters_summary
from neuralgym.ops.summary_ops import scalar_summary, images_summary

from function import bbox2mask, local_patch, resize_z_like
from function import gen_conv, gen_deconv, dis_conv, e_conv, gen_conv_add_z,_residual_block
from function import resize_mask_like, contextual_attention
import tensorflow as tf

logger = logging.getLogger()

class PIIGANModel(Model):
    def __init__(self):
        super().__init__('PIIGANModel')

    def build_inpaint_net(self, x_rgb, z, config = None, reuse = False,
                          training = True, padding = 'SAME', name = 'inpaint_net'):
        """Inpaint network.

        Args:
            x: tensor [-1, 1] -> 3 channels
            z: latent vector
        Returns:
            [-1, 1] as predicted feature tensor (8 channels)
        """
        # z = z * mask # masking the noise ?

        # cnum = 32
        # Encode the image
        w = x_rgb.get_shape()[1]

        # reshape to image size 1 channel
        z = tf.reshape(z, [-1, w, w, 1], name = 'z_reshape')

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            mul, mul_var = self.build_landmark_encoder(x_rgb, z, reuse=reuse)
            muf, muf_var = self.build_face_region_encoder(x_rgb, z, reuse=reuse)

            zl = self.reparameterize(mul, mul_var)
            zf = self.reparameterize(muf, muf_var)

            print("sampled z shapes", zl.shape, zf.shape)

            z_emb = tf.concat([zf, zl], axis=-1, name = 'z_emb')
            print("z_emb shape", z_emb.shape)

            landmarks = self.build_landmark_decoder(zl, w, reuse=reuse)
            face_mask = self.build_mask_decoder(z_emb, w, reuse=reuse)
            face_part = self.build_face_decoder(z_emb, w, reuse=reuse)

            print("z_emb shape", z_emb.shape)

            rec = self.build_gan(x_rgb, z_emb, reuse=reuse)
            rec = tf.concat([x_rgb, landmarks, face_mask, face_part], axis=3, name = 'x_false')

            return rec, muf, muf_var, mul, mul_var

    def build_gan(self, x_masked, z_latent,  reuse = False, training = True):
        cnum = 32
        with tf.compat.v1.variable_scope("gan_fcn", reuse=reuse):
            # replace with image size / tf.layers.convi
            x = x_masked
            print(":::::::::::::  GAN  ::::::::::::::")
            print("x shape", x.shape)
            print("z_latent gan shape", z_latent.shape)
            z = tf.reshape(z_latent, [-1, 8,8, 8])
            print("z_latent gan reshaped shape", z.shape)

            x = tf.compat.v1.layers.conv2d(x, cnum, 3, 1, name='conv1', padding='SAME')	
            x = c1 = tf.compat.v1.layers.conv2d(x, cnum,3, 1,  name='conv2', padding='SAME') # 128
            x = tf.compat.v1.layers.conv2d(x, cnum *2,3, 2,  name='conv3', padding='SAME')# 64)# 64
            x = c2 = tf.compat.v1.layers.conv2d(x, cnum *2,3, 1,  name='conv4', padding='SAME')# 64) 
            x = tf.compat.v1.layers.conv2d(x, cnum *4,3, 2,  name='conv5', padding='SAME')
            x = c3 = tf.compat.v1.layers.conv2d(x, cnum *4,3, 1,  name='conv6', padding='SAME') # 32
            x = tf.compat.v1.layers.conv2d(x, cnum *8,3, 2, name='conv7', padding='SAME') # 16
            x = c4 = tf.compat.v1.layers.conv2d(x, cnum *8,3, 1,  name='conv8', padding='SAME')
            x = tf.compat.v1.layers.conv2d(x, cnum *16,3, 2,  name='conv9', padding='SAME') # 8
            x = tf.compat.v1.layers.conv2d(x, cnum *16,3, 1,  name='conv10', padding='SAME')
            # x = c2 = gen_conv(x, 3, 1, cnum *2) 
            # x = gen_conv(x, 3, 2, cnum *4)
            # x = c3 = gen_conv(x, 3, 1, cnum *4) # 32
            # x = gen_conv(x, 3, 2, cnum *8) # 16
            # x = c4 = gen_conv(x, 3, 1, cnum *8)
            # x = gen_conv(x, 3, 2, cnum *16) # 8
            # x = gen_conv(x, 3, 1, cnum *16)
            print('gan x before reshape', x.shape)
            x = tf.concat([x,z], axis = -1, name='concat1')
            x = tf.compat.v1.layers.conv2d_transpose(x, cnum * 8,3, 2,  name='deconv1', padding='SAME')
            x = tf.concat([x, c4], axis = 3, name='concat2')
            x = tf.compat.v1.layers.conv2d(x, cnum *8,3, 1,  name='conv11', padding='SAME')
            x = tf.compat.v1.layers.conv2d_transpose(x, cnum * 4,3, 2,  name='deconv2', padding='SAME')
            x = tf.concat([x, c3], axis = 3, name='concat3')
            x = tf.compat.v1.layers.conv2d(x, cnum * 4,3, 1,  name='conv12', padding='SAME')
            x = tf.compat.v1.layers.conv2d_transpose(x, cnum * 2,3, 2,  name='deconv3', padding='SAME')
            x = tf.concat([x, c2], axis = 3, name='concat4')
            x = tf.compat.v1.layers.conv2d(x, cnum * 2,3, 1,  name='conv13', padding='SAME')
            x = tf.compat.v1.layers.conv2d_transpose(x, cnum,3, 2,  name='deconv4' , padding='SAME')
            x = tf.concat([x, c1], axis = 3, name='concat5')
            x = tf.compat.v1.layers.conv2d(x, cnum ,3, 1,  name='conv14', padding='SAME')
            x = tf.compat.v1.layers.conv2d(x, 3,3, 1,  activation=tf.nn.tanh, name= 'conv15', padding='SAME')

            return x

    def reparameterize(self, mu, sigma):
        eps = tf.random.normal(shape = tf.shape(input=mu))
        std = mu + tf.exp(sigma / 2) * eps

        z = tf.add(mu, tf.multiply(std, eps))
        return z

    def build_landmark_encoder(self, x, z_rgb_masked, reuse = False, training = True):
        with tf.compat.v1.variable_scope("landmark_encoder", reuse=reuse):
            z_shape_down1 = x.get_shape().as_list()[1] // 2
            print("shape down", z_shape_down1)
            print("shape ", z_rgb_masked.shape)
            z_feature_down1 = resize_z_like(z_rgb_masked, [z_shape_down1, z_shape_down1])
            print("z_feature_down1", z_feature_down1.shape)
            z_feature_down2 = resize_z_like(z_rgb_masked, [z_shape_down1 // 2, z_shape_down1 // 2])
            print("z_feature_down2", z_feature_down2.shape)
            z_feature_down3 = resize_z_like(z_rgb_masked, [z_shape_down1 // 4, z_shape_down1 // 4])
            z_feature_down4 = resize_z_like(z_rgb_masked, [z_shape_down1 // 8, z_shape_down1 // 8])
            z_feature_down5 = resize_z_like(z_rgb_masked, [z_shape_down1 // 16, z_shape_down1 // 16])
            cnum = 32
            x = gen_conv_add_z(x, z_feature_down1, cnum, 4, 2, name='conv1_downsample')
            x = gen_conv_add_z(x, z_feature_down2, cnum * 2, 4, 2, name='conv2_downsample')
            x = gen_conv_add_z(x, z_feature_down3, cnum * 4, 4, 2, name='conv3_dowmsample')
            x = gen_conv_add_z(x, z_feature_down4, cnum * 4, 4, 2, name='conv4_downsample')
            x = gen_conv_add_z(x, z_feature_down5, cnum * 4, 4, 2, name='conv5_downsample')
            print("pre flatten shape", x.shape)
            x = flatten(x, name='flatten')
            zl = tf.compat.v1.layers.dense(x, 256, name = 'zl')
            zl_var = tf.compat.v1.layers.dense(x, 256, name = 'zl_var')
            print("landmark encoder output shape", zl.shape)
            return zl, zl_var

    def build_face_region_encoder(self, x, z_rgb_masked, reuse=False, training = True):
        with tf.compat.v1.variable_scope("face_mask_encoder", reuse=reuse):
            ones_x = tf.ones_like(x)[:, :, :, 0:1]
            z_shape_down1 = x.get_shape().as_list()[1] // 2
            z_feature_down1 = resize_z_like(z_rgb_masked, [z_shape_down1, z_shape_down1])
            z_feature_down2 = resize_z_like(z_rgb_masked, [z_shape_down1 // 2, z_shape_down1 // 2])
            z_feature_down3 = resize_z_like(z_rgb_masked, [z_shape_down1 // 4, z_shape_down1 // 4])
            z_feature_down4 = resize_z_like(z_rgb_masked, [z_shape_down1 // 8, z_shape_down1 // 8])
            z_feature_down5 = resize_z_like(z_rgb_masked, [z_shape_down1 // 16, z_shape_down1 // 16])
            cnum = 32
            x = gen_conv_add_z(x, z_feature_down1, cnum, 4, 2, name='conv1_downsample')
            x = gen_conv_add_z(x, z_feature_down2, cnum * 2, 4, 2, name='conv2_downsample')
            x = gen_conv_add_z(x, z_feature_down3, cnum * 4, 4, 2, name='conv3_dowmsample')
            x = gen_conv_add_z(x, z_feature_down4, cnum * 4, 4, 2, name='conv4_downsample')
            x = gen_conv_add_z(x, z_feature_down5, cnum * 4, 4, 2, name='conv5_downsample')
            x = flatten(x, name='flatten')
            w = x.get_shape()[1]
            zf = tf.compat.v1.layers.dense(x, 256, name = 'zf')
            zf_var = tf.compat.v1.layers.dense(x, 256, name = 'zf_var')
            return zf, zf_var 

    def build_mask_decoder(self, z, w, reuse = False, training = True):
        with tf.compat.v1.variable_scope("face_mask_decoder", reuse=reuse):
            cnum = 32
            z_shape_down = w // 16
            c = z.shape[1] // (z_shape_down * z_shape_down)
            z = tf.reshape(z, [-1, z_shape_down, z_shape_down, c])
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 8, 4, 2, name = 'deconv1', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 4, 4, 2, name = 'deconv2', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 2, 4, 2, name = 'deconv3', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum, 4, 2, name = 'deconv4', padding='SAME')
            z = tf.compat.v1.layers.conv2d(z, 1, 3, 1, activation=tf.nn.tanh, name='conv1', padding='SAME')
            return z

    def build_face_decoder(self, z, w, reuse = False, training = True):
        with tf.compat.v1.variable_scope("face_decoder", reuse=reuse):
            cnum = 32
            z_shape_down = w // 16
            c = z.shape[1] // (z_shape_down * z_shape_down)
            z = tf.reshape(z, [-1, z_shape_down, z_shape_down, c])
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 8, 4, 2, name = 'deconv1', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 4, 4, 2, name = 'deconv2', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 2, 4, 2, name = 'deconv3', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum, 4, 2, name = 'deconv4', padding='SAME')
            z = tf.compat.v1.layers.conv2d(z, 3, 3, 1, activation=tf.nn.tanh, name='conv1', padding='SAME')
            return z

    def build_landmark_decoder(self, z, w, reuse = False, training = True):
        with tf.compat.v1.variable_scope("landmark_decoder", reuse=reuse):
            cnum = 32
            z_shape_down = w // 16
            print("zl decoder shape", z.shape)
            print("z_shape_down", z_shape_down)
            c = z.shape[1] // (z_shape_down * z_shape_down)
            print("z shape decoder", z.shape)
            z = tf.reshape(z, [-1, z_shape_down, z_shape_down, c])
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 8, 4, 2, name = 'deconv1', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 4, 4, 2, name = 'deconv2', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum * 2, 4, 2, name = 'deconv3', padding='SAME')
            z = tf.compat.v1.layers.conv2d_transpose(z, cnum, 4, 2, name = 'deconv4', padding='SAME')
            z = tf.compat.v1.layers.conv2d(z, 1, 3, 1, activation=tf.nn.tanh, name='conv1', padding='SAME')
            return z

    def build_wgan_local_discriminator(self, x, reuse = False, training = True):
        with tf.compat.v1.variable_scope('discriminator_local', reuse = reuse):
            cnum = 64
            x = dis_conv(x, cnum, name = 'conv1', training = training)
            x = dis_conv(x, cnum * 2, name = 'conv2', training = training)
            x = dis_conv(x, cnum * 4, name = 'conv3', training = training)
            x = dis_conv(x, cnum * 8, name = 'conv4', training = training)
            x = flatten(x, name = 'flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse = False, training = True):
        with tf.compat.v1.variable_scope('discriminator_global', reuse = reuse):
            cnum = 64
            x = dis_conv(x, cnum, name = 'conv1', training = training)
            x = dis_conv(x, cnum * 2, name = 'conv2', training = training)
            x = dis_conv(x, cnum * 4, name = 'conv3', training = training)
            x = dis_conv(x, cnum * 4, name = 'conv4', training = training)
            x = flatten(x, name = 'flatten')
            return x

    # out shape batch , 64 * 4 * w /16 * w /16 , x = o chanlles
    def build_extractor(self, x, reuse = False, training = True):
        with tf.compat.v1.variable_scope('build_extractor', reuse = reuse):
            cnum = 64
            w = x.get_shape()[1]
            x = e_conv(x, cnum, name = 'conv1', training = training)
            x = e_conv(x, cnum * 2, name = 'conv2', training = training)
            x = e_conv(x, cnum * 4, name = 'conv3', training = training)
            x = e_conv(x, cnum * 4, name = 'conv4', training = training)
            x = flatten(x, name = 'flatten')
            print("extracto x shape pre dense", x.shape)
            z = tf.compat.v1.layers.dense(x, w * w, name = 'z')
            z_var = tf.compat.v1.layers.dense(x, w * w, name = 'z_var')
            print("extractor oout shape", z.shape)
            return z, z_var

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse = False, training = True):
        with tf.compat.v1.variable_scope('discriminator', reuse = reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse = reuse, training = training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse = reuse, training = training)
            dout_local = tf.compat.v1.layers.dense(dlocal, 1, name = 'dout_local_fc')
            dout_global = tf.compat.v1.layers.dense(dglobal, 1, name = 'dout_global_fc')

            return dout_local, dout_global

    def build_encode_z(self, batch_local):
        mu, logvar = self.build_extractor(batch_local, reuse = tf.compat.v1.AUTO_REUSE, training = True)

        eps = tf.random.normal(shape = tf.shape(input=mu))
        std = mu + tf.exp(logvar / 2) * eps

        z = tf.add(mu, tf.multiply(std, eps))

        return mu, logvar, z

    def build_graph_with_losses(self, batch_data, config, training = True,
                                summary = True, reuse = False):
        # batch of original images are being normalized to [-1, 1]
        batch_pos = batch_data / 127.5 - 1.
        batch_norm_rgb = batch_pos[:, :, :, 0:3]
        # generate mask, 1 represents masked point
#         bbox = random_bbox_np(config)
#         bbox = (32, 32, 64, 64)
        bbox = (tf.constant(config.HEIGHT // 2), tf.constant(config.WIDTH // 2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))

        mask = bbox2mask(bbox, config, name = 'mask_c')  # masked:1
        print("mask shape: centered in image)", mask.shape)	

        # batch masked removing the part of mask 
        batch_incomplete = batch_pos * (1. - mask)
        batch_incomplete_rgb = batch_incomplete[:, :, :, 0:3]
        batch_landmarks = batch_pos[:, :, :, 3]
        batch_mask = batch_pos[:, :, :, 4]
        batch_part = batch_pos[:, :, :, 5]

        # creating random noise for input at the network
        z = tf.random.normal(shape = [batch_incomplete.get_shape()[0].value, batch_incomplete.get_shape()[1].value, batch_incomplete.get_shape()[2].value, 1])
        print("z shape", z.shape)
        # Excute the network with random nose and incomplete images -> step 2 in paper
        ft_false, *_ = self.build_inpaint_net(
            batch_incomplete_rgb, z, config, reuse = reuse, training = training,
            padding = config.PADDING)
        losses = {}
        batch_predicted = ft_false
        fake_rgb = ft_false[:, :, :, 0:3]

        # apply mask and complete image(2, 256, 256, 3) / complelting the image with the incomplete image from previous network
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        batch_complete_rgb = batch_complete[:, :, :, 0:3]
        
        fake_landmarks = ft_false[:, :, :, 3]
        fake_mask = ft_false[:, :, :, 4]
        fake_part = ft_false[:, :, :, 5]

        l1_alpha = config.COARSE_L1_ALPHA

        landmark_scaled_fake, landmark_scaled_true = (fake_landmarks + 1.) / 2., (batch_landmarks + 1.) / 2.
        mask_scaled_fake, mask_scaled_true = (fake_mask + 1.) / 2., (batch_mask + 1.) / 2.
        ce_landmarks = - ( landmark_scaled_true * tf.math.log(landmark_scaled_fake + 1e-8) + (1. - landmark_scaled_true) * tf.math.log(1. - landmark_scaled_fake + 1e-8) )
        ce_mask = - ( mask_scaled_true * tf.math.log(mask_scaled_fake + 1e-8) + (1. - mask_scaled_true) * tf.math.log(1. - mask_scaled_fake + 1e-8) )
        # calculate cross enytropy loss for mask and landmaarks, l1 loss for part
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(input_tensor=ce_landmarks)
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(input_tensor=ce_mask)
        losses['ae_loss'] += l1_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_part - fake_part))

        # local patches(2, 128, 128, 3)
        # looks like taking patches from original images
        local_patch_batch_pos = local_patch(batch_norm_rgb, bbox)
        print("local_patch_batch_pos shape", local_patch_batch_pos.shape)
        # Now takling patches from predicted image
        local_patch_batch_complete = local_patch(batch_complete_rgb, bbox)
        # Now taking patches from mask for gradients penalty showing 
        local_patch_mask = local_patch(mask, bbox)

        # loss lloks like is calculating the loss of predicted feature tensor and masking
        # this should be the Consistency loss for the generator
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_norm_rgb - fake_rgb) * (1. - mask))    
        losses['ae_loss'] /= tf.reduce_mean(input_tensor=1. - mask)
        if summary:
#             scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete, ft_false]
            print("batch pos shape", batch_pos.shape)
            print("batch incomplete shape", batch_incomplete.shape)
            print("batch complete shape", batch_complete.shape)
            print("rec shape", ft_false.shape)
            # i predict that every one of them has 8 channels 13, 128, 128, 8
            # crop first 3 channels
            for i in range(len(viz_img)):
                viz_img[i] = viz_img[i][:, :, :, 0:3]
            # if offset_flow is not None:
            #     viz_img.append(
            #         resize(offset_flow, scale = 4,
            #                func = tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis = 0),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # batch of original and predicted masked images, probably for training the discriminator
        batch_pos_neg = tf.concat([batch_norm_rgb, batch_complete_rgb], axis = 0)
        # local deterministic patch, patches from original images and complelted images, probably for the local discriminator 
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE * 2, 1, 1, 1])], axis = 3)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training = training, reuse = reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # regress z loss
            print("local patch batch pos neg shape", local_patch_batch_pos_neg.shape)

            # Extractor encodes the batch of local patches, why? 
            mu, logvar, z_encode = self.build_encode_z(local_patch_batch_pos_neg)
            # spliting mu and logvar of the extractor

            # This process looks like extractor thing 
            mu_real, mu_fake = tf.split(mu, 2)
            logvar_real, _ = tf.split(logvar, 2)
            z_real, _ = tf.split(z_encode, 2)
            local_patch_z = local_patch(z, bbox)
            z_real = tf.reshape(z_real, local_patch_z.get_shape())
            padding = [[0, 0], [bbox[0], bbox[0]], [bbox[1], bbox[1]], [0, 0]]
#             padding = [[0, 0], [32, 32], [32, 32], [0, 0]]
            z_real = tf.pad(tensor=z_real, paddings=padding)
            z_fake_label = flatten(local_patch_z, 'z_fake_label')

            # This looks like the l1 loss between the random vector and the vector from the extractor
            losses['l1_regress_z_loss'] = config.REGRESSION_Z_LOSS_ALPHA * tf.reduce_mean(input_tensor=tf.abs(mu_fake - z_fake_label))
            scalar_summary('losses/l1_regress_z_loss', losses['l1_regress_z_loss'])
            print("z_real_shape", z_real.shape)
            print("padding ", config.PADDING)

            # Now the network is being executed using the vector from the extractor 
            rec_real, *_ = self.build_inpaint_net(
                batch_incomplete_rgb, z_real, config = config, reuse = tf.compat.v1.AUTO_REUSE, training = training,
                padding = config.PADDING)

            z_real_batch_predicted = rec_real
            z_real_batch_predicted_rgb = z_real_batch_predicted[:, :, :, 0:3]
            # apply mask and complete image(2, 256, 256, 3)
            z_real_batch_complete = z_real_batch_predicted_rgb * mask + batch_incomplete_rgb * (1. - mask)
            # local patches(2, 128, 128, 3)
            z_real_local_patch_x1 = local_patch(z_real_batch_predicted_rgb, bbox)
            losses['l1_loss_z_x'] = l1_alpha * tf.reduce_mean(input_tensor=tf.abs(local_patch_batch_pos - z_real_local_patch_x1))
            # if not config.PRETRAIN_COARSE_NETWORK:
                # losses['l1_loss_z_x'] += l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - z_real_local_patch_x2))
            scalar_summary('losses/l1_loss_z_x', losses['l1_loss_z_x'])
            losses['ae_loss_z_x'] = l1_alpha * tf.reduce_mean(input_tensor=tf.abs(batch_pos - rec_real) * (1. - mask))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['ae_loss_z_x'] += tf.reduce_mean(input_tensor=tf.abs(batch_norm_rgb - z_real_batch_predicted_rgb) * (1. - mask))
            losses['ae_loss_z_x'] /= tf.reduce_mean(input_tensor=1. - mask)
            scalar_summary('losses/ae_loss_z_x', losses['ae_loss_z_x'])
            losses['loss_kl'] = -0.5 * tf.reduce_sum(input_tensor=1 + logvar_real - tf.square(mu_real) - tf.exp(logvar_real))
            scalar_summary('losses/loss_kl', losses['loss_kl'])

            # viz_img_res = [batch_pos, batch_incomplete, z_real_batch_complete, z_real_batch_predicted]
            # images_summary(
            #     tf.concat(viz_img_res, axis = 0),
            #     'res', config.VIZ_MAX_OUT)

            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name = 'gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name = 'gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_norm_rgb, batch_complete_rgb)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse = True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask = local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask = mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name = 'g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name = 'g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name = 'g_loss')
            gradients_summary(losses['g_loss'], ft_false, name = 'g_loss_to_x1')
            gradients_summary(losses['g_loss'], ft_false, name = 'g_loss_to_x2')
            #gradients_summary(losses['l1_loss'], x1, name = 'l1_loss_to_x1')
            #gradients_summary(losses['l1_loss'], x2, name = 'l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], ft_false, name = 'ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], ft_false, name = 'ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
#         losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] += config.LOSS_KL * losses['loss_kl']
        losses['g_loss'] += config.REGRESSION_Z_LOSS_ALPHA * losses['l1_loss_z_x']
        losses['g_loss'] += config.REGRESSION_Z_LOSS_ALPHA * losses['ae_loss_z_x']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        if config.REGRESSION_Z_LOSS:
            losses['g_loss'] += config.REGRESSION_Z_LOSS_ALPHA * losses['l1_regress_z_loss']
            logger.info('Set REGRESSION_Z_LOSS_ALPHA to %f' % config.REGRESSION_Z_LOSS_ALPHA)
        g_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        # cross entropy loss between xf and xm
        # add losses from the hvae
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox = None, name = 'val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
#           bbox = random_bbox_np(config)
          bbox = (tf.constant(config.HEIGHT // 2), tf.constant(config.WIDTH // 2),
                  tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        mask = bbox2mask(bbox, config, name = name + 'mask_c')
        batch_pos = batch_data / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - mask)
        # inpaint
        z = tf.random.normal(shape = [batch_incomplete.get_shape()[0].value, batch_incomplete.get_shape()[1].value, batch_incomplete.get_shape()[2].value, 1])
        rec, *_ = self.build_inpaint_net(
            batch_incomplete, z, config, reuse = True,
            training = False, padding = config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = rec
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = rec
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        # global image visualization
        # viz_img = [batch_pos, batch_incomplete, batch_complete]
        # if offset_flow is not None:
        #     viz_img.append(
        #         resize(offset_flow, scale = 4,
        #                func = tf.image.resize_nearest_neighbor))
        # images_summary(
            # tf.concat(viz_img, axis = 2),
            # name + '_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT // 2), tf.constant(config.WIDTH // 2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name)

    def build_server_graph(self, batch_data, reuse = False, is_training = False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis = 2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        z = tf.random.normal(shape = [batch_incomplete.get_shape()[0].value, batch_incomplete.get_shape()[1].value, batch_incomplete.get_shape()[2].value, 1])
        rec,  _ = self.build_inpaint_net(
            batch_incomplete, z, reuse = reuse, training = is_training,
            config = None)
        batch_predict = rec
        # apply mask and reconstruct
        batch_complete = batch_predict * masks + batch_incomplete * (1 - masks)
        return batch_complete

    def build_server_graph__(self, batch_data, reuse = False, is_training = False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks =  tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        bbox = (tf.constant(64 // 2), tf.constant(64 // 2),
        tf.constant(64), tf.constant(64))

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        #inpaint
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        mu, logvar, z_encode = self.build_encode_z(local_patch_batch_pos)

        z_encode =  tf.reshape(z_encode, (34, 64, 64, 1))
        padding = [[0, 0], [32, 32], [52, 32], [0, 0]]
        z_encode = tf.pad(tensor=z_encode, paddings=padding)

        return z_encode
