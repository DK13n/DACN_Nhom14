from src.Model_arch.train_mode import FASHead, train_model

from src.Model_arch import CDRes_ViTModel as CDResV

import torch

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fusion_CDResV = CDResV.CDRes_ViT(device,d_model=256)

    model = FASHead(fusion_CDResV,device,256,2)

    train_model(model,device,r"src\Model_arch\Data\metadata\metadata\publics_train_metadata.csv",
            r"src\Model_arch\Data\publics_data_train",20,1e-3,5e-4,out_dir=r"Model",exp_name="checkpoint")
