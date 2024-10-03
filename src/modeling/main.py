import sys

from omegaconf import OmegaConf

from modeling.data import load_data
from modeling.model import initialize_model
from modeling.trainer import Trainer


def main():
    config = OmegaConf.load(sys.argv[1])

    trainer = Trainer(config, **config.trainer_config)
    trainer.save_config(config)

    tokenizer, (train_dataset, val_dataset, test_dataset) = load_data(
        **config.data_config
    )
    model = initialize_model(tokenizer, **config.model_config)

    trainer.setup_dataloaders(
        tokenizer,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    trainer.setup_model(model)

    trainer.train()
    trainer.test()

    trainer.cleanup()


if __name__ == "__main__":
    main()
