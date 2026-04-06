import numpy as np
import json
import os
from datetime import datetime
from huggingface_hub import HfApi


class ModelRegistry:  # Define a class to manage model versions and storage

    def __init__(self, repo_id, local_dir='registry'):  # Constructor runs when object is created
        self.repo_id = repo_id  # Store Hugging Face repo ID (where models will be uploaded)
        self.local_dir = local_dir  # Store local directory name (default = "registry")
        self.api = HfApi()  # Create Hugging Face API object to interact with HF Hub

        # Create path: registry/models → where model files will be stored locally
        self.models_dir = os.path.join(local_dir, 'models')

        # Create path: registry/registry.json → file to store version info
        self.registry_path = os.path.join(local_dir, 'registry.json')


        # Create models directory if it does not exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Check if registry.json file already exists
        if os.path.exists(self.registry_path):

            # Open the file in read mode
            with open(self.registry_path, 'r') as f:

                # Load JSON data into Python dictionary
                self.registry = json.load(f)

        else:
            # If registry file does not exist → create new empty registry
            self.registry = {
                'versions': [],  # List to store model versions
                'current_production': None  # Track which version is in production (None = not set yet)
            }

            # Save this new registry to registry.json file
            self._save_registry()

    def _save_registry(self):
        # Open registry.json in write mode
        with open(self.registry_path, 'w') as f:

            # Write the registry dictionary into JSON file (pretty format with indent=2)
            json.dump(self.registry, f, indent=2)



    def register_model(self, weights, config, metrics):
        # Figure out the next version number
        # if we have 0 versions then next will be v1 and if we have meny others it'll be just n...n
        version_num = len(self.registry['versions']) + 1
        version = f'v{version_num}'

        # Create a folder for new versions
        version_dir = os.path.join(self.models_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # Save the three files into the version folders
        # 1. Weights (the trained parameters)
        weights_path = os.path.join(version_dir, 'weights.npz')
        np.savez(weights_path, **weights)

        # Save the Hyperparams as well we need to for self Improvement loops
        # 2. Configs (Hyperparams)
        config_path = os.path.join(version_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # to see how well model performed in on tests
        # 3. Metrics (how well the model performed)
        metrics_path = os.path.join(version_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create the version entry for registry.json
        entry = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'status': 'candidate',
            'accuracy': metrics.get('mean_accuracy', 0),
            'config': config,
        }

        # Add the version history to save
        self.registry['versions'].append(entry)
        self._save_registry()

        # Upload all three files to HF hub
        for filename in ['weights.npz', 'config.json', 'metrics.json']:
            filepath = os.path.join(version_dir, filename)
            self.api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo = f'models/{version}/{filename}',
                repo_id = self.repo_id,
                repo_type = 'dataset',
            )

        # Upload registry.json too
        self.api.upload_file(
            path_or_fileobj=self.registry_path,
            path_in_repo='registry.json',
            repo_id=self.repo_id,
            repo_type='dataset'
        )

        print(f'  [Registry] Registered {version} (accuracy={entry["accuracy"]:.4f})')
        return version

    # Helper Fuc^n
    def _upload_registry(self):
        self.api.upload_file(
            path_or_fileobj=self.registry_path,
            path_in_repo='registry.json',
            repo_id=self.repo_id,
            repo_type='dataset'
        )


    def promote(self, version):
        # Find the version entry in the registry
        target = None
        for v in self.registry['versions']:
            if v['version'] == version:
                target = v
                break

        if target is None:
            print(f'  [Registry] No such version {version}')
            return False

        # Check if there's a current production models
        current_prod = self.registry['current_production']

        if current_prod is None:
            target['status'] = 'production'
            self.registry['current_production'] = version
            self._save_registry()
            self._upload_registry()
            print(f'  [Registry] {version} promoted to production (first model)')
            return True

        # Find the current production model's entry
        prod_entry = None
        for v in self.registry['versions']:
            if v['version'] == current_prod:
                prod_entry = v
                break

        # Compare the acc w.r.t other models or we can say previous
        if target['accuracy'] > prod_entry['accuracy']:
            # New model wins. Promote it, retire the old one.
            prod_entry['status'] = 'retired'
            target['status'] = 'production'
            self.registry['current_production'] = version
            self._save_registry()
            self._upload_registry()
            print(f'  [Registry] {version} promoted to production '
                  f'({target["accuracy"]:.4f} > {prod_entry["accuracy"]:.4f})')
            return True
        else:
            # Old model wins. Keep it.
            print(f'  [Registry] {version} stays candidate '
                  f'({target["accuracy"]:.4f} <= {prod_entry["accuracy"]:.4f})')
            return False


    def get_production_model(self):
        # Check if any model in the production
        current = self.registry['current_production']
        if current is None:
            print(f'  [Registry] No current production model')
            return None, None

        # Load weights from the production version's folder
        version_dir = os.path.join(self.models_dir, current)
        weights_path = os.path.join(version_dir, 'weights.npz')
        config_path = os.path.join(version_dir, 'config.json')

        weights = dict(np.load(weights_path))

        with open(config_path, 'r') as f:
            config = json.load(f)

        return weights, config

    def list_versions(self):
        if not self.registry['versions']:
            print('  No versions registered.')
            return

        print(f"\n{'Version':<10} {'Status':<12} {'Accuracy':<10} {'Timestamp'}")
        print("-" * 55)
        for v in self.registry['versions']:
            print(f"{v['version']:<10} {v['status']:<12} {v['accuracy']:<10.4f} {v['timestamp'][:19]}")
        print(f"\nProduction: {self.registry['current_production'] or 'None'}")

if __name__ == '__main__':
    import sys

    repo_id = 'rekhi/deadneurons-registry'
    registry = ModelRegistry(repo_id)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.registry.registry list")
        print("  python -m src.registry.registry promote <version>")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'list':
        registry.list_versions()
    elif cmd == 'promote' and len(sys.argv) >= 3:
        registry.promote(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")