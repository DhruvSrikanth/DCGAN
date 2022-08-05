config = {
    'device' : {
        'home directory' : './',
        'initial seed': 1, 
        'device type': 'cpu', 
        
        }, 
    'data' : {
        'batch size': 64,
        'num workers': 8, 
        'image shape': (1, 64, 64),
        'train samples': 60000, 
        'test samples': 4000
        }, 
    'hyperparameters' : {
        'latent dimension': 100,
        'learning rate': 0.0002, 
        'beta1': 0.5,
        'beta2': 0.999,
        'epochs': 200, 
        'discriminator epochs': 1,
        'generator epochs': 1,
    }, 
    'save' : {
        'sample interval': 1,
        'sample save path': 'samples',
        'model save path': 'weights'
    }, 
    'log' : {
        'log path': 'logs',
        'experiment number': 1
    }
}
