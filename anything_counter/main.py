import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path='../config', config_name='config', version_base='1.2.0')
def run(config: DictConfig) -> None:
    instantiate(config['anything_counter']).run()


if __name__ == '__main__':
    run()
