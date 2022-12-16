import torch

BATCH_SIZE = 4
RESIZE_TO = 512
NUM_EPOCHS = 1000

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

BASE_DIR = '/backend'
# trainin images and XML files dir
TRAIN_DIR = F'{BASE_DIR}/train'
# validation image directory (test image)
VALID_DIR = f'{BASE_DIR}/test'

CLASSES = ['background', 'I love you', 'Thank You', 'Yes']

NUM_CLASSES = 4

VISUALIZE_TRANSORMED_IMAGES = False

# location to dave model and plost
OUT_DIR = f'{BASE_DIR}/outputs'
SAVE_PLOTS_EPOCH = 2  # SAVE LOSS PLOT AFTER EVERY 2 EPOCHS
SAVE_MODEL_EPOCH = 2

print("successfully create config")
print('DEVICE:', DEVICE)
