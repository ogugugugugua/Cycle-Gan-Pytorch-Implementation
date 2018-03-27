import model
import data_loader
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import itertools
import torchvision

#DATA
a_loader,b_loader,a_test_loader,b_test_loader = data_loader.get_loader()

#GPU
gpu_ids = [0]
gpu_id = [str(i) for i in gpu_ids]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)

#model
assert (torch.cuda.is_available())
Da = model.D(device_ids=gpu_ids).cuda()
Db = model.D(device_ids=gpu_ids).cuda()
Ga = model.G(device_ids=gpu_ids).cuda()
Gb = model.G(device_ids=gpu_ids).cuda()

#address
ckpt_dir = './checkpoints/celeba_cyclegan'
data_loader.mkdir(ckpt_dir)

#criterion
GANloss = nn.MSELoss()
L1 = nn.L1Loss()
lr = 0.0002
epochs = 50
da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))

#test data
a_real_test = Variable(iter(a_test_loader).next()[0], volatile=True).cuda(gpu_ids[0],async=True)
b_real_test = Variable(iter(b_test_loader).next()[0], volatile=True).cuda(gpu_ids[0],async=True)

for epoch in range(epochs):
    for i, (a_real, b_real) in enumerate(itertools.izip(a_loader, b_loader)):

#----------------------Train G-----------------------#
        #set train
        Ga.train()
        Gb.train()

        #get data
        a_real = Variable(a_real[0]).cuda(gpu_ids[0],async=True)
        b_real = Variable(b_real[0]).cuda(gpu_ids[0],async=True)

        #generate 4 images
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)
        a_reconstruct = Ga(b_fake)
        b_reconstruct = Gb(a_fake)

        #get score from D
        a_fake_score = Da(a_fake)
        b_fake_score = Db(b_fake)

        #compute gan loss for generated fake image
        one_label = Variable(torch.ones(a_fake_score.size())).cuda(gpu_ids[0],async=True)
        a_gan_loss = GANloss(a_fake_score,one_label)
        b_gan_loss = GANloss(b_fake_score,one_label)

        #compute reconstruction loss between generated fake image and reconstructed image
        a_rec_loss = L1(a_reconstruct,a_real)
        b_rec_loss = L1(b_reconstruct,b_real)

        g_loss_sum = a_gan_loss + a_rec_loss + b_gan_loss + b_rec_loss

        #update parameters of G
        Ga.zero_grad()
        Gb.zero_grad()
        g_loss_sum.backward()
        ga_optimizer.step()
        gb_optimizer.step()


#----------------------Train D-----------------------#

        a_fake_pool = data_loader.ItemPool()
        b_fake_pool = data_loader.ItemPool()
        a_fake1 = Variable(torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0])).cuda(gpu_ids[0],async=True)
        b_fake1 = Variable(torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0])).cuda(gpu_ids[0],async=True)

        #get score from D
        a_real_score = Da(a_real)
        b_real_score = Db(b_real)
        a_fake1_score = Da(a_fake1)
        b_fake1_score = Db(b_fake1)
        one_label = Variable(torch.ones(a_fake1_score.size())).cuda(gpu_ids[0],async=True)
        zero_label = Variable(torch.zeros(a_fake1_score.size())).cuda(gpu_ids[0],async=True)

        #compute loss
        a_d_r_loss = GANloss(a_real_score, one_label)
        b_d_r_loss = GANloss(b_real_score, one_label)
        a_d_f_loss = GANloss(a_fake1_score, zero_label)
        b_d_f_loss = GANloss(b_fake1_score, zero_label)

        a_d_loss = a_d_r_loss + a_d_f_loss
        b_d_loss = b_d_r_loss + b_d_f_loss

        #update parameters of D
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()

        #print result of training
        if (i + 1) % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, min(len(a_loader), len(b_loader))))

#----------------------Evaluation-----------------------#
        if (i + 1) % 100 == 0:
            Ga.eval()
            Gb.eval()

            # train G
            a_fake_test = Ga(b_real_test)
            b_fake_test = Gb(a_real_test)

            a_rec_test = Ga(b_fake_test)
            b_rec_test = Gb(a_fake_test)

            pic = (
                  torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1) / 2.0

            save_dir = './sample_images_while_training'
            data_loader.mkdir(save_dir)
            torchvision.utils.save_image(pic,
                                     '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, min(len(a_loader), len(b_loader))),
                                     nrow=3)

    data_loader.save_checkpoint({'epoch': epoch + 1,
                                   'Da': Da.state_dict(),
                                   'Db': Db.state_dict(),
                                   'Ga': Ga.state_dict(),
                                   'Gb': Gb.state_dict(),
                                   'da_optimizer': da_optimizer.state_dict(),
                                   'db_optimizer': db_optimizer.state_dict(),
                                   'ga_optimizer': ga_optimizer.state_dict(),
                                   'gb_optimizer': gb_optimizer.state_dict()},
                                  '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                                  max_keep=2)
