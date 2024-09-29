import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from gp.lightning.metric import EvalKit
from gp.utils.utils import dict_res_summary, load_pretrained_state


def lightning_fit(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    num_epochs,
    profiler=None,
    cktp_prefix="",
    load_best=True,
    prog_freq=20,
    test_rep=1,
    save_model=True,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    reload_freq=0,
    val_interval=None,
    check_n_epoch=1,
    strategy=None,
    grad_acc_step=1,
    grad_clipping=None,
    save_step=None,
    save_epoch=None,
    save_time=None,
    precision="32-True",
    top_k=1,
    ckpt_path=None,
    save_last=False,
    ckpt_save_path=None,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    if save_model:
        if save_epoch is not None:
            callbacks.append(ModelCheckpoint(dirpath=ckpt_save_path, monitor=metrics.val_metric, mode=metrics.eval_mode, save_last=True,
                filename=cktp_prefix + "{epoch}-{step}", every_n_epochs=save_epoch, ))
        else:
            callbacks.append(
                ModelCheckpoint(dirpath=ckpt_save_path, save_last=save_last, filename=cktp_prefix + "{epoch}-{step}", train_time_interval=save_time,
                    every_n_epochs=save_epoch, every_n_train_steps=save_step, save_top_k=top_k))

    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        # devices=1 if torch.cuda.is_available() else 10,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        precision=precision,
        enable_checkpointing=save_model,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
        reload_dataloaders_every_n_epochs=reload_freq,
        check_val_every_n_epoch=check_n_epoch, num_sanity_val_steps=1,
        gradient_clip_val=grad_clipping,
        accumulate_grad_batches=grad_acc_step, val_check_interval=val_interval
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    model.model.save_partial(model.model.save_dir + "/save_ckpt.pth")
    if load_best:
        model_dir = trainer.checkpoint_callback.best_model_path
        deep_speed = False
        if strategy[:9] == "deepspeed":
            deep_speed = True
        state_dict = load_pretrained_state(model_dir, deep_speed)
        model.load_state_dict(state_dict)


    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]


def lightning_test(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    model_dir: str = None,
    strategy="auto",
    profiler=None,
    prog_freq=20,
    test_rep=1,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
    )
    if model_dir:
        deep_speed = False
        if strategy[:9] == "deepspeed":
            deep_speed = True
        state_dict = load_pretrained_state(model_dir, deep_speed)
        model.load_state_dict(state_dict)


    val_col = []
    for i in range(test_rep):
        val_col.append(
            trainer.validate(model, datamodule=data_module, verbose=False)[0]
        )

    val_res = dict_res_summary(val_col)
    for met in val_res:
        val_mean = np.mean(val_res[met])
        val_std = np.std(val_res[met])
        print("{}:{:f}±{:f}".format(met, val_mean, val_std))

    target_val_mean = np.mean(val_res[metrics.val_metric])
    target_val_std = np.std(val_res[metrics.val_metric])

    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    return [target_val_mean, target_val_std], [
        target_test_mean,
        target_test_std,
    ]
