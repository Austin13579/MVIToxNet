import torch
import torch.nn as nn
import torch.nn.functional as F


class MVIToxNet(torch.nn.Module):
    def __init__(self, embed_dim=128):
        super(MVIToxNet, self).__init__()
        self.char_embedding = nn.Embedding(64, embed_dim)
        self.bpe_embedding = nn.Embedding(498, embed_dim)
        self.atom_embedding = nn.Embedding(82, embed_dim)

        # SMILES CNN
        # Character-level Conv
        self.char_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.LayerNorm(80 - 2),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            )

        # BPE-level Conv
        self.bpe_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.LayerNorm(40 - 2),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            )
        
        # Atom-level Conv
        self.atom_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            nn.Softplus(),
            nn.LayerNorm(60 - 2),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3),
            )


        # Fingerprint Encoder
        self.fp_encoder = nn.Sequential(
            nn.Linear(167+ 2048 , 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
        )

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128 * 1, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )

    def forward(self, fp, seq1, seq2,seq3):
        ## Fingerprint embedding
        fp_emb = self.fp_encoder(fp)

        ## SMILES embedding
        # Atom-level
        a_emb = self.atom_embedding(seq2)
        a_emb=self.atom_conv(a_emb.transpose(2, 1))
        a_emb=self.max_pool(a_emb).squeeze()

        # Char-level
        c_emb = self.char_embedding(seq1)
        c_emb = c_emb.transpose(2, 1)
        c_emb = self.char_conv(c_emb)
        c_emb = self.max_pool(c_emb).squeeze()

        # BPE-level
        b_emb = self.bpe_embedding(seq3)
        b_emb = b_emb.transpose(2, 1)
        b_emb = self.bpe_conv(b_emb)
        b_emb = self.max_pool(b_emb).squeeze()

        # Fusion
        rep=a_emb +0.1*b_emb+0.1*c_emb+0.1*fp_emb
        out = self.decoder(rep)
        return out

