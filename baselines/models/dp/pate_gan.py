# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pate_gan.py implements the PATE_GAN generative model to generate private synthetic data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import math
from models.dp.utils.architectures import Generator, Discriminator
from models.dp.utils.helper import weights_init, pate, moments_acc


class PATE_GAN:
    def __init__(
        self,
        input_dim,
        z_dim,
        num_teachers,
        target_epsilon,
        target_delta,
        conditional=True,
    ):
        self.generator = Generator(z_dim, input_dim, conditional).cuda().double()
        self.student_disc = Discriminator(input_dim, wasserstein=False).cuda().double()
        self.teacher_disc = [
            Discriminator(input_dim, wasserstein=False).cuda().double()
            for _ in range(num_teachers)
        ]
        self.generator.apply(weights_init)
        self.student_disc.apply(weights_init)
        self.z_dim = z_dim
        self.num_teachers = num_teachers
        for i in range(num_teachers):
            self.teacher_disc[i].apply(weights_init)

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional

    def train(self, x_train, y_train, hyperparams):
        batch_size = hyperparams.batch_size
        num_teacher_iters = hyperparams.num_teacher_iters
        num_student_iters = hyperparams.num_student_iters
        num_moments = hyperparams.num_moments
        lap_scale = hyperparams.lap_scale
        class_ratios = None
        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)

        real_label = 1
        fake_label = 0

        alpha = torch.cuda.DoubleTensor([0.0 for _ in range(num_moments)])
        l_list = 1 + torch.cuda.DoubleTensor(range(num_moments))

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=hyperparams.lr)
        optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=hyperparams.lr)
        optimizer_td = [
            optim.Adam(self.teacher_disc[i].parameters(), lr=hyperparams.lr)
            for i in range(self.num_teachers)
        ]

        tensor_data = data_utils.TensorDataset(
            torch.cuda.DoubleTensor(x_train), torch.cuda.DoubleTensor(y_train)
        )

        train_loader = []
        for teacher_id in range(self.num_teachers):
            start_id = teacher_id * len(tensor_data) / self.num_teachers
            end_id = (
                (teacher_id + 1) * len(tensor_data) / self.num_teachers
                if teacher_id != (self.num_teachers - 1)
                else len(tensor_data)
            )

            train_loader.append(
                data_utils.DataLoader(
                    torch.utils.data.Subset(
                        tensor_data, range(int(start_id), int(end_id))
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )

        steps = 0
        epsilon = 0

        while epsilon < self.target_epsilon:

            # train the teacher discriminators
            for t_2 in range(num_teacher_iters):
                for i in range(self.num_teachers):
                    inputs, categories = None, None
                    for b, data in enumerate(train_loader[i], 0):
                        inputs, categories = data
                        break

                    # train with real
                    optimizer_td[i].zero_grad()
                    label = torch.full((inputs.size()[0],), real_label).cuda()
                    output = self.teacher_disc[i].forward(
                        torch.cat([inputs, categories.unsqueeze(1).double()], dim=1)
                    )
                    label = label.unsqueeze(1)
                    label = label.double()
                    err_d_real = criterion(output, label)
                    err_d_real.backward()

                    # train with fake
                    z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
                    label.fill_(fake_label)

                    if self.conditional:
                        category = (
                            torch.multinomial(
                                class_ratios, inputs.size()[0], replacement=True
                            )
                            .unsqueeze(1)
                            .cuda()
                            .double()
                        )
                        fake = self.generator(torch.cat([z.double(), category], dim=1))
                        output = self.teacher_disc[i].forward(
                            torch.cat([fake.detach(), category], dim=1)
                        )
                    else:
                        fake = self.generator(z.double())
                        output = self.teacher_disc[i].forward(fake)

                    err_d_fake = criterion(output, label.double())
                    err_d_fake.backward()
                    optimizer_td[i].step()

            # train the student discriminator
            for t_3 in range(num_student_iters):
                z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()

                if self.conditional:
                    category = (
                        torch.multinomial(
                            class_ratios, inputs.size()[0], replacement=True
                        )
                        .unsqueeze(1)
                        .cuda()
                        .double()
                    )
                    fake = self.generator(torch.cat([z.double(), category], dim=1))
                    predictions, clean_votes = pate(
                        torch.cat([fake.detach(), category], dim=1),
                        self.teacher_disc,
                        lap_scale,
                    )
                    outputs = self.student_disc.forward(
                        torch.cat([fake.detach(), category], dim=1)
                    )
                else:
                    fake = self.generator(z.double())
                    predictions, clean_votes = pate(
                        fake.detach(), self.teacher_disc, lap_scale
                    )
                    outputs = self.student_disc.forward(fake.detach())

                # update the moments
                alpha = alpha + moments_acc(
                    self.num_teachers, clean_votes, lap_scale, l_list
                )

                # update student

                err_sd = criterion(outputs, predictions)

                optimizer_sd.zero_grad()
                err_sd.backward()
                optimizer_sd.step()

            # train the generator
            optimizer_g.zero_grad()
            z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
            label = torch.full((inputs.size()[0],), real_label).cuda()

            if self.conditional:
                category = (
                    torch.multinomial(class_ratios, inputs.size()[0], replacement=True)
                    .unsqueeze(1)
                    .cuda()
                    .double()
                )
                fake = self.generator(torch.cat([z.double(), category], dim=1))
                output = self.student_disc(torch.cat([fake, category.double()], dim=1))
            else:
                fake = self.generator(z.double())
                output = self.student_disc.forward(fake)
            label = label.unsqueeze(1)
            label = label.double()
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()

            # Calculate the current privacy cost
            epsilon = min((alpha - math.log(self.target_delta)) / l_list)
            if steps % 100 == 0:
                print(
                    "Step : ",
                    steps,
                    "Loss SD : ",
                    err_sd.item(),
                    "Loss G : ",
                    err_g.item(),
                    "Epsilon : ",
                    epsilon.item(),
                )

            steps += 1

    def generate(self, num_rows, class_ratios=None, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if self.conditional:
                cat = (
                    torch.multinomial(class_ratios, batch_size, replacement=True)
                    .unsqueeze(1)
                    .cuda()
                    .double()
                )
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)

            else:
                synthetic = self.generator(noise.double())

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps * batch_size < num_rows:
            noise = torch.randn(num_rows - steps * batch_size, self.z_dim).cuda()

            if self.conditional:
                cat = (
                    torch.multinomial(
                        class_ratios, num_rows - steps * batch_size, replacement=True
                    )
                    .unsqueeze(1)
                    .cuda()
                    .double()
                )
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)
            else:
                synthetic = self.generator(noise.double())
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)


def train_pate_gan(data_path, save_path, cat_cols, eps):
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    import pandas as pd

    # Loading the data
    raw_data = pd.read_csv(data_path)

    # 1. 识别列类型
    data_columns = raw_data.columns.tolist()
    if cat_cols == "all":
        cat_cols = data_columns
    num_cols = [c for c in data_columns if c not in cat_cols]

    # 2. 对分类特征进行独热编码 (One-Hot Encoding)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if len(cat_cols) > 0:
        cat_data = ohe.fit_transform(raw_data[cat_cols])
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        # 合并数值列和独热编码列
        train_raw = np.hstack([raw_data[num_cols].values, cat_data])
        current_feature_names = num_cols + cat_feature_names
    else:
        train_raw = raw_data[num_cols].values
        current_feature_names = num_cols

    # 3. 使用 MinMaxScaler 将所有特征缩放到 [0, 1]
    scaler = MinMaxScaler()
    train = scaler.fit_transform(np.nan_to_num(train_raw))
    X_train = train[:, :-1]
    y_train = train[:, -1]

    class Hyperparams:
        batch_size = 64
        num_teacher_iters = 5
        num_student_iters = 5
        num_moments = 100
        lap_scale = 0.0001 #0.01
        lr = 1e-4

    hyperparams = Hyperparams()

    input_dim = X_train.shape[1]
    z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)
    num_teachers = X_train.shape[0] // 1000

    # PATE-GAN requires GPU
    if torch.cuda.is_available():
        pate_gan = PATE_GAN(
            input_dim, z_dim, num_teachers, eps, 1e-5, conditional=False
        )
        pate_gan.train(X_train, y_train, hyperparams)

        syn_data = pate_gan.generate(X_train.shape[0])
        # 4. 逆变换：首先还原 MinMaxScaler 的缩放
        # syn_data 的最后一列是 target_col，前面是特征

        syn_features_raw = scaler.inverse_transform(syn_data)
        syn_df_features = pd.DataFrame(syn_features_raw, columns=current_feature_names)

        # 5. 逆变换：还原独热编码为原始类别标签
        if len(cat_cols) > 0:
            # ohe.inverse_transform 会自动在每组独热列中执行 argmax
            decoded_cats = ohe.inverse_transform(
                syn_df_features[cat_feature_names].values
            )

            final_df = syn_df_features[num_cols].copy()
            final_df[cat_cols] = decoded_cats
        else:
            final_df = syn_df_features.copy()

        # 6. 恢复原始列顺序并保存
        final_df = final_df[data_columns]
        final_df.to_csv(save_path, index=False)
        print(f"Saved PATE-GAN data to {save_path}")
    else:
        print("PATE-GAN requires CUDA, but it is not available.")
