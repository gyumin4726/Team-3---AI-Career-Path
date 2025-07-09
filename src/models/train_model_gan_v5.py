import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import click
import torch.nn as nn
import torch.optim as optim
import logging
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path
import sys
from argparse import Namespace
from src.data.dataset import TEPCSVDataset, CSVToTensor, CSVNormalize, InverseNormalize
from src.models.utils import get_latest_model_id
from src.models.recurrent_models import TEPRNN, LSTMGenerator
from src.models.convolutional_models import (
    CNN1D2DDiscriminatorMultitask
)
import torch.backends.cudnn as cudnn
from src.data.dataset import TEPDataset
from src.models.utils import time_series_to_plot
from tensorboardX import SummaryWriter
import random
from PIL import Image

"""This is training of Conditioned GAN model script."""

REAL_LABEL = 1
FAKE_LABEL = 0


@click.command()
@click.option('--cuda', required=True, type=int, default=7)
@click.option('--run_tag', required=True, type=str, default="unknown")
@click.option('--random_seed', required=False, type=int, default=42)
def main(cuda, run_tag, random_seed):
    """
    GAN v5 모델 훈련 - CSV 데이터 사용
    
    Args:
        cuda: GPU 번호
        run_tag: 실험 태그 (로그 구분용)
        random_seed: 랜덤 시드 (옵션)
    """
    # for tensorboard logs
    try:
        os.makedirs("logs")
    except OSError:
        pass

    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO)
    latest_model_id = get_latest_model_id(dir_name="models") + 1
    prefix = f'{latest_model_id}_{run_tag}_tmp_'
    temp_model_dir = TemporaryDirectory(dir="models", prefix=prefix)
    temp_model_dir.cleanup()
    Path(temp_model_dir.name).mkdir(parents=True, exist_ok=False)
    Path(os.path.join(temp_model_dir.name), "images").mkdir(parents=True, exist_ok=False)
    Path(os.path.join(temp_model_dir.name), "weights").mkdir(parents=True, exist_ok=False)
    temp_model_dir_tensorboard = TemporaryDirectory(dir="logs", prefix=prefix)
    temp_model_dir_tensorboard.cleanup()
    Path(temp_model_dir_tensorboard.name).mkdir(parents=True, exist_ok=False)
    temp_log_file = os.path.join(temp_model_dir.name, 'log.txt')
    file_handler = logging.FileHandler(temp_log_file)
    file_handler.setFormatter(log_formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Training begin on {device}')
    logger.info(f'Tmp TB dir {temp_model_dir_tensorboard.name}, tmp model dir {temp_model_dir.name}')

    with open(__file__, 'r') as f:
        with open(os.path.join(temp_model_dir.name, "script.py"), 'w') as out:
            print("# This file was saved automatically during the experiment run.\n", end='', file=out)
            for line in f.readlines():
                print(line, end='', file=out)


    logger.info(f"Random Seed: {random_seed}")
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = True

    lstm_size = 64
    loader_jobs = 4
    window_size = 30
    bs = 128
    # CSV 파일 경로 설정 (항상 CSV 사용)
    train_csv_dir = "data/train_faults"
    train_csv_files = [os.path.join(train_csv_dir, f"train_fault_{i}.csv") for i in range(13)]
    noise_size = 100
    conditioning_size = 1
    in_dim = noise_size + conditioning_size
    fault_type_classifier_weights = "models/2/latest.pth"
    checkpoint_every = 10
    real_fake_w_d = 1.0  # weight for real fake in loss
    fault_type_w_d = 0.8  # weight for fault type term in loss
    real_fake_w_g = 1.0  # weight for real fake in loss
    similarity_w_g = 1.0  # weight for fault type term in loss
    generator_train_prob = 0.8  # how frequently ignore the generator step. Note that the amount of total steps will be
    # equal to epochs * 83 batches * generator_train_prob, which is (1 - generator_train_prob) % less that for
    # discriminator
    epochs = 300

    # Create writer for tensorboard
    writer = SummaryWriter(temp_model_dir_tensorboard.name)
    # todo: add all running options to some structure and print them.
    writer.add_text('Options', str("Here yo can write running options or put your ads."), 0)

    # 데이터 변환 설정 (항상 CSV 사용)
    transform = transforms.Compose([
        CSVToTensor(),
        CSVNormalize()
    ])

    inverse_transform = InverseNormalize()

    logger.info("Preparing dataset...")
    # 항상 CSV 파일 사용
    trainset = TEPCSVDataset(
        csv_files=train_csv_files,
        transform=transform,
        is_test=False
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=loader_jobs,
                                              drop_last=False)

    logger.info("Dataset done.")

    logger.info("Preparing print dataset...")
    # 항상 CSV 파일 사용 (시각화용)
    printset = TEPCSVDataset(
        csv_files=train_csv_files,
        transform=None,  # 정규화 없이 원본 데이터 사용
        is_test=False
    )
    print_batch_size = min(trainset.class_count, 21)  # 클래스 개수만큼, 최대 21개

    printloader = torch.utils.data.DataLoader(printset, batch_size=print_batch_size, shuffle=False, num_workers=1,
                                              drop_last=False)

    logger.info("Dataset done.")
 
    netG = LSTMGenerator(in_dim=in_dim, out_dim=52, hidden_dim=256, n_layers=4).to(device)

    netD = CNN1D2DDiscriminatorMultitask(input_size=trainset.features_count, n_layers_1d=4, n_layers_2d=4,
                                         n_channel=trainset.features_count * 3, n_channel_2d=40,
                                         class_count=trainset.class_count,
                                         kernel_size=9, dropout=0.2, groups=trainset.features_count).to(device)

    logger.info("Generator:\n" + str(netG))
    logger.info("Discriminator:\n" + str(netD))

    binary_criterion = nn.BCEWithLogitsLoss()
    cross_entropy_criterion = nn.CrossEntropyLoss()
    similarity = nn.MSELoss(reduction='mean')


    optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

    logger.info("Models done.")
    for epoch in range(epochs):

        logger.info('Epoch %d training...' % epoch)
        netD.train()
        netG.train()
        # can do this cause we dont use optimizer for this net now

        for i, data in enumerate(trainloader, 0):
            n_iter = epoch * len(trainloader) + i

            real_inputs, fault_labels = data["shot"], data["label"]
            real_inputs, fault_labels = real_inputs.to(device), fault_labels.to(device)
            real_inputs = real_inputs.squeeze(dim=1)
            fault_labels = fault_labels.squeeze()

            netD.zero_grad()

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # Real data training
            batch_size, seq_len = real_inputs.size(0), real_inputs.size(1)
            real_target = torch.full((batch_size, seq_len, 1), REAL_LABEL, dtype=torch.float32, device=device)
            # for  label smoothing on [0.74, 1.0]
            # real_target = (0.74 - 1.0) - torch.rand(batch_size, seq_len, 1, device=device) + 1.0

            type_logits, fake_logits = netD(real_inputs, None)
            errD_real = binary_criterion(fake_logits, real_target)
            errD_type_real = cross_entropy_criterion(type_logits.transpose(1, 2), fault_labels)
            # todo: print at tensorboard errD_real + errD_type

            errD_complex_real = real_fake_w_d * errD_real + fault_type_w_d * errD_type_real
            errD_complex_real.backward()
            # here is missed sigmoid function
            D_x = type_logits.mean().item()

            # Fake data training
            noise = torch.randn(batch_size, seq_len, noise_size, device=device)
            random_labels = torch.randint(high=trainset.class_count, size=(batch_size, 1, 1),
                                          dtype=torch.float32, device=device)
            random_labels = random_labels.repeat(1, seq_len, 1)
            random_labels[:, :20, :] = 0
            noise = torch.cat((noise, random_labels), dim=2)

            state_h, state_c = netG.zero_state(batch_size)
            state_h, state_c = state_h.to(device), state_c.to(device)

            fake_inputs = netG(noise, (state_h, state_c))
            fake_target = torch.full((batch_size, seq_len, 1), FAKE_LABEL, dtype=torch.float32, device=device)
            # for  label smoothing on [0.0, 0.3]
            # fake_target = (0.0 - 0.3) - torch.rand(batch_size, seq_len, 1, device=device) + 0.3
            # WARNING: do not forget about detach!
            type_logits, fake_logits = netD(fake_inputs.detach(), None)
            errD_fake = binary_criterion(fake_logits, fake_target)
            errD_type_fake = cross_entropy_criterion(type_logits.transpose(1, 2), random_labels.long().squeeze())

            errD_complex_fake = real_fake_w_d * errD_fake + fault_type_w_d * errD_type_fake
            errD_complex_fake.backward()
            # here is missed sigmoid function
            D_G_z1 = type_logits.mean().item()
            # all good here (2 lines)
            errD = real_fake_w_d * errD_real + real_fake_w_d * errD_fake
            errD_fault_type = fault_type_w_d * errD_type_real + fault_type_w_d * errD_type_fake

            optimizerD.step()

            # Visualize discriminator gradients
            for name, param in netD.named_parameters():
                writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, n_iter)

            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            g_coin = random.random() < generator_train_prob
            errG = torch.zeros(1)
            D_G_z2 = torch.zeros(1).item()
            errG_similarity = torch.zeros(1)

            if g_coin:
                netG.zero_grad()

                type_logits, fake_logits = netD(fake_inputs, None)
                errG = real_fake_w_g * binary_criterion(fake_logits, real_target)
                errG.backward()
                D_G_z2 = fake_logits.mean().item()

                # Visual similarity correction
                # todo: add KL-divergence term
                noise = torch.randn(batch_size, seq_len, noise_size, device=device)
                # fault_labels의 shape를 확인하고 적절히 처리
                if fault_labels.numel() == batch_size:
                    # 1D 라벨인 경우 시퀀스 길이로 확장
                    fault_labels_expanded = fault_labels.float().view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
                elif fault_labels.numel() == batch_size * seq_len:
                    # 이미 시퀀스 길이를 포함한 경우
                    fault_labels_expanded = fault_labels.float().view(batch_size, seq_len, 1)
                else:
                    raise ValueError(f"Unexpected fault_labels shape: {fault_labels.shape}")
                noise = torch.cat((noise, fault_labels_expanded), dim=2)

                state_h, state_c = netG.zero_state(batch_size)
                state_h, state_c = state_h.to(device), state_c.to(device)
                out_seqs = netG(noise, (state_h, state_c))

                errG_similarity = similarity_w_g * similarity(out_seqs, real_inputs)
                errG_similarity.backward()

                optimizerG.step()

            # Visualize generator gradients
            for name, param in netG.named_parameters():
                writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, n_iter)

            log_flag = True
            if log_flag:
                logger.info('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' %
                            (epoch, epochs, i, len(trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            writer.add_scalar('GeneratorLoss', errG.item(), n_iter)
            writer.add_scalar('GeneratorLossSimilarity', errG_similarity.item(), n_iter)

            writer.add_scalar('DiscriminatorLoss', errD.item(), n_iter)
            writer.add_scalar('DiscriminatorLossFaultType', errD_fault_type.item(), n_iter)
            writer.add_scalar('DofX_noSigmoid', D_x, n_iter)
            writer.add_scalar('DofGofz_noSigmoid', D_G_z1, n_iter)

        logger.info('Epoch %d passed' % epoch)
        # trainset.shuffle()

        # Saving epoch results.
        # The following images savings cost a lot of memory, reduce the frequency
        if epoch in [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, *list(range(60, epochs, 30)), epochs - 1]:
            netD.eval()
            netG.eval()

            real_display = next(iter(printloader))
            real_inputs, true_labels = real_display["shot"], real_display["label"]
            real_inputs, true_labels = real_inputs.to(device), true_labels.to(device)

            real_display = inverse_transform(real_display)

            real_plots = time_series_to_plot(real_display["shot"].cpu())
            for idx, rp in enumerate(real_plots):
                fp_real = os.path.join(temp_model_dir.name, "images", f"{epoch}_epoch_real_{idx}.jpg")
                ndarr = rp.to('cpu', torch.uint8).permute(1, 2, 0).numpy()
                im = Image.fromarray(ndarr, mode="RGB")
                im.save(fp_real, format=None)
                writer.add_image(f"RealTEP_{idx}", rp, epoch)

            batch_size, seq_len = real_inputs.size(0), real_inputs.size(1)
            noise = torch.randn(batch_size, seq_len, noise_size, device=device)
            # true_labels의 shape를 확인하고 적절히 처리
            if true_labels.numel() == batch_size:
                # 1D 라벨인 경우 시퀀스 길이로 확장
                true_labels_expanded = true_labels.float().view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
            elif true_labels.numel() == batch_size * seq_len:
                # 이미 시퀀스 길이를 포함한 경우
                true_labels_expanded = true_labels.float().view(batch_size, seq_len, 1)
            else:
                raise ValueError(f"Unexpected true_labels shape: {true_labels.shape}")
            noise = torch.cat((noise, true_labels_expanded), dim=2)

            state_h, state_c = netG.zero_state(batch_size)
            state_h, state_c = state_h.to(device), state_c.to(device)
            fake_display = netG(noise, (state_h, state_c))
            fake_display = {"shot": fake_display.cpu(), "label": true_labels}
            fake_display = inverse_transform(fake_display)

            fake_plots = time_series_to_plot(fake_display["shot"])
            for idx, fp in enumerate(fake_plots):
                fp_fake = os.path.join(temp_model_dir.name, "images", f"{epoch}_epoch_fake_{idx}.jpg")
                ndarr = fp.to('cpu', torch.uint8).permute(1, 2, 0).numpy()
                im = Image.fromarray(ndarr, mode="RGB")
                im.save(fp_fake, format=None)
                writer.add_image(f"FakeTEP_{idx}", fp, epoch)

        if (epoch % checkpoint_every == 0) or (epoch == (epochs - 1)):
            torch.save(netG, os.path.join(temp_model_dir.name, "weights", f"{epoch}_epoch_generator.pth"))
            torch.save(netD, os.path.join(temp_model_dir.name, "weights", f"{epoch}_epoch_discriminator.pth"))

        # change_print_sim_run은 TEPDatasetV4에만 있음 (CSV 모드에서는 불필요)
        pass

    logger.info(f'Finished training for {epochs} epochs.')

    file_handler.close()
    writer.close()

    os.rename(
        temp_model_dir.name,
        os.path.join("models", f'{latest_model_id}_{run_tag}')
    )

    os.rename(
        temp_model_dir_tensorboard.name,
        os.path.join("logs", f'{latest_model_id}_{run_tag}')
    )


if __name__ == '__main__':
    main()
