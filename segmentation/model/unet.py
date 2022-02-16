#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:41:45 2020

@author: mibook

UNet Model Class

This code holds the defination for u-net model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Single Encoder Block

    Transforms large image with small inchannels into smaller image with larger
    outchannels, via two convolution / relu pairs.
    """
    def __init__(self, inchannels, outchannels, dropout, spatial, kernel_size=3, padding=1):
        super().__init__()
        self.outchannels = outchannels
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=kernel_size, padding=padding)
        if dropout > 0:
            if spatial:
                self.dropout = nn.Dropout2d(p=dropout)
            else:
                self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        x = self.dropout(x)
        return x


class UpBlock(nn.Module):
    """
    Single Decoder Block

    Transforms small image with large inchannels into larger image with smaller
    outchannels, via two convolution / relu pairs.
    """
    def __init__(self, inchannels, outchannels, dropout, spatial, kernel_size=2, stride=2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=kernel_size, stride=stride
        )
        if dropout > 0:
            self.conv = ConvBlock(inchannels, outchannels, dropout, spatial)

    def forward(self, x, skips):
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class Unet(nn.Module):
    """
    U-Net Model

    Combines the encoder and decoder blocks with skip connections, to arrive at
    a U-Net model.
    """
    def __init__(self, inchannels, outchannels, net_depth, dropout = 0.2, spatial = False, first_channel_output=16):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_channels = inchannels
        out_channels = first_channel_output
        for _ in range(net_depth):
            conv = ConvBlock(in_channels, out_channels, dropout, spatial)
            self.downblocks.append(conv)
            in_channels, out_channels = out_channels, 2 * out_channels

        self.middle_conv = ConvBlock(in_channels, out_channels, dropout, spatial)

        in_channels, out_channels = out_channels, int(out_channels / 2)
        for _ in range(net_depth):
            upconv = UpBlock(in_channels, out_channels, dropout, spatial)
            self.upblocks.append(upconv)
            in_channels, out_channels = out_channels, int(out_channels / 2)

        self.seg_layer = nn.Conv2d(2 * out_channels, outchannels, kernel_size=1)

    def forward(self, x):
        decoder_outputs = []

        for layer in self.downblocks:
            decoder_outputs.append(layer(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for layer in self.upblocks:
            x = layer(x, decoder_outputs.pop())
        return self.seg_layer(x)