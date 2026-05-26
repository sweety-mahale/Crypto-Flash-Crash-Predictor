import os
import yaml

class Config:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml') # fallback if config in root
            
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = {}

    @property
    def raw_data_path(self):
        return self._config.get('data', {}).get('raw_data_path', 'data/raw/')

    @property
    def processed_data_path(self):
        return self._config.get('data', {}).get('processed_data_path', 'data/processed/')

    @property
    def symbols(self):
        return self._config.get('data', {}).get('symbols', ['BTCUSDT'])

    @property
    def crash_threshold(self):
        return self._config.get('features', {}).get('crash_definition', {}).get('threshold', -0.03)

    @property
    def crash_window(self):
        return self._config.get('features', {}).get('crash_definition', {}).get('window', 300)

    @property
    def arf_params(self):
        return self._config.get('model', {}).get('params', {'n_models': 10, 'max_features': 'sqrt', 'lambda_value': 6})

    @property
    def drift_delta(self):
        return self._config.get('model', {}).get('drift_detector', {}).get('delta', 0.002)

    @property
    def api_host(self):
        return self._config.get('api', {}).get('host', '0.0.0.0')

    @property
    def api_port(self):
        return self._config.get('api', {}).get('port', 8000)

config = Config()
