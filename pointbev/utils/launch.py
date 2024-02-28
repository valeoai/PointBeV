def modif_config_based_on_flags(config):
    """Modify the configuration file based on flags.

    Available flags:
        - debug: use mini dataset and debug mode.
        - val_sparse: use sparse validation mode.
    """
    debug = config.flags.debug
    val_sparse = config.flags.val_sparse

    if debug:
        config.data.version = "mini"

        config.train = True
        config.data.batch_size = 1
        config.test = False
        config.data.valid_batch_size = 1
        config.trainer.num_sanity_val_steps = 1
        config.trainer.overfit_batches = 1

        config.data.train_shuffle = True
        config.model.train_kwargs.train_visu_imgs = True
        config.model.val_kwargs.val_visu_imgs = True
        config.model.name = "debug"
        config.trainer.max_epochs = 1000
        config.trainer.check_val_every_n_epoch = 20
        config.model.train_kwargs.train_visu_epoch_frequency = 10
        config.trainer.log_every_n_steps = 1
        config.trainer.strategy = "ddp"

    if val_sparse:
        config.model.net.sampled_kwargs.val_mode = "regular_pillars"
        config.model.net.sampled_kwargs.patch_size = 1
        config.model.net.sampled_kwargs.valid_fine = True
        config.model.net.sampled_kwargs.N_coarse = 2000
        config.model.net.sampled_kwargs.N_fine = "dyna"
        config.model.net.sampled_kwargs.N_anchor = "dyna"
        config.model.net.sampled_kwargs.fine_thresh = 0.1
        config.model.net.sampled_kwargs.fine_patch_size = 9
        config.data.valid_batch_size = 1
