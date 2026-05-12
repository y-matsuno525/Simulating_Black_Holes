from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS, load_config
from simulation import configure, run_simulation


def main():
    config = load_config()
    configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)
    run_simulation()


if __name__ == "__main__":
    main()
