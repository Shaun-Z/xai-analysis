"""
Training script that loads configuration from YAML file
"""

import argparse
import yaml
import sys
from train import train_model


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """Convert config dictionary to argparse namespace"""
    args = argparse.Namespace()
    
    # Set all configuration values as attributes
    for key, value in config.items():
        setattr(args, key.replace('-', '_'), value)
    
    return args


def main():
    parser = argparse.ArgumentParser(description='Train model from YAML configuration')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                       help='Override config values (format: key=value)')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(cmd_args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{cmd_args.config}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    
    # Apply overrides
    for override in cmd_args.override:
        if '=' not in override:
            print(f"Invalid override format: {override}. Use key=value format.")
            sys.exit(1)
        
        key, value = override.split('=', 1)
        
        # Try to convert to appropriate type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
        
        config[key] = value
        print(f"Override: {key} = {value}")
    
    # Convert to args and start training
    args = config_to_args(config)
    train_model(args)


if __name__ == '__main__':
    main()