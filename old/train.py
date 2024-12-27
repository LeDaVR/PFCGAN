import os
import logging
import shutil
import socket
import time

from model import PIIGANModel
import neuralgym as ng # pip install git+https://github.com/JiahuiYu/neuralgym
import tensorflow as tf
from PFCData import PFCData

import numpy as np


logger = logging.getLogger()


def multigpu_graph_def(model, data, config, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(config.BATCH_SIZE)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            images, config, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            images, config, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')

if __name__ == "__main__":
    config = ng.Config('setting.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    # training data
    with open(config.DATA_FLIST['celeba']['train']['flist']) as f:
        fnames = f.read().splitlines()
    data = PFCData(
        fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP, preprocessed_dir = config.DATA_FLIST['celeba']['train']['preprocessed'])
    images = data.data_pipeline(config.BATCH_SIZE)
    #images have the shape of (13, 128, 128, 3)

    print("images shape:", images.get_shape())
    # Add face kepoints channel(1), face mask(1), and face part(3) -> total of 8 channels
    # use the 'celeba' to look for images with  the postfix _kepoints.jpg, _mask.jpg, and _part.jpg


    # main model
    model = PIIGANModel()
    print(":::::::::::::::::::: PFCGAN :::::::::::::::::::::::::::::")
    g_vars, d_vars, losses = model.build_graph_with_losses(
        images, config=config)
    print(":::::::::::::::::::: PFCGAN :::::::::::::::::::::::::::::")
    # validation images
    if config.VAL:
        with open(config.DATA_FLIST['celeba']['validation']['flist']) as f:	
            val_fnames = f.read().splitlines()
        # progress monitor by visualizing static images
        for i in range(config.STATIC_VIEW_SIZE):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                static_images, config, name='static_view/%d' % i)
    # training settings
    lr = tf.compat.v1.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.compat.v1.constant_initializer(1e-4))
    d_optimizer = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    g_optimizer = d_optimizer
    # gradient processor
    if config.GRADIENT_CLIP:
        gradient_processor = lambda grad_var: (
            tf.compat.v1.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
            grad_var[1])
    else:
        gradient_processor = None
    # log dir
    log_prefix = config.MODEL_LOG + '_'.join([
        socket.gethostname(), 'celeba']) + '_' + str(time.strftime("%m-%d_%H%M"))
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=5,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'd'},
    )
    # train generator with primary trainer
    trainer = ng.train.Trainer(
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=config.MAX_ITERS,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'g'},
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
    )
    # add all callbacks
    if not config.PRETRAIN_COARSE_NETWORK:
        trainer.add_callbacks(discriminator_training_callback)
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.compat.v1.summary.merge_all()),
    ])
    # launch training
    shutil.copy('./model.py',log_prefix+'/')
    trainer.train()
