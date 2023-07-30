import yaml

def cfg_from_yaml(config_path):
    with open(config_path,'r') as f:
        cfg = yaml.load(f.read())
    return cfg
    

if __name__ == "__main__":
    cfg = cfg_from_yaml('cfgs/train.yaml')
    print(type(cfg))
    print(cfg)