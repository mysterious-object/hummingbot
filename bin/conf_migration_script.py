import argparse

import path_util  # noqa: F401

from chimerabot.client.config.conf_migration import migrate_configs
from chimerabot.client.config.config_crypt import ETHKeyFileSecretManger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate the ChimeraBot confs")
    parser.add_argument("password", type=str, help="Required to migrate all encrypted configs.")
    args = parser.parse_args()
    secrets_manager_ = ETHKeyFileSecretManger(args.password)
    migrate_configs(secrets_manager_)
