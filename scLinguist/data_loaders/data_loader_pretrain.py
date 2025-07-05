import json
import os
import random
import re

import pyarrow.parquet as pq

import scanpy
import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
# from muon import MuData

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import bernoulli, beta
from typing import Optional, Iterable, Tuple, Union
from numbers import Integral, Real
from warnings import warn

from scipy.sparse import issparse, csc_matrix, csr_matrix
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from anndata import AnnData
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
# LOCAL_RANK = 0
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))


def max_min_normalization_with_nan(data):
    High = 1
    Low = 1e-8
    scale = High - Low

    min_vals = np.nanmin(data, axis=1)
    max_vals = np.nanmax(data, axis=1)

    data = data.astype(float)
    nan_mask = np.isnan(data)

    for i in range(data.shape[0]):
        if np.all(nan_mask[i]):
            continue  # 整列都是NaN值，保留为NaN
        data[i] -= min_vals[i]
        if max_vals[i] != min_vals[i]:
            data[i] /= (max_vals[i] - min_vals[i])
        data[i] *= scale
        data[i] += Low

    return data


class scMultiDataset(Dataset):
    def __init__(self, data_dir_1, data_dir_2):
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        # self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
        (
            self.data_array,
            self.len_array,
            self.data_array_1,
            self.len_array_1,
            self.length,
        ) = self.load_all_data()

    def __len__(self):
        return self.length

    def load_all_data(self):
        data_list = []
        len_list = []
        length = 0

        # for path in self.files:
        for path in [
            self.data_dir_1
        ]:
            adata = scanpy.read(path)
            scanpy.pp.normalize_total(
                adata,
                target_sum=10000
            )
            scanpy.pp.log1p(adata)
            data = adata.X
            data_list.append(data)

            length += adata.shape[0]
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)

        data_list = []
        mask_list = []
        len_list = []
        length = 0

        # for path in self.files:
        for path in [
            self.data_dir_2
        ]:
            adata = scanpy.read(path)
            # clr(adata)
            data = adata.X.todense()

            data = max_min_normalization_with_nan(data)
            # data = np.where(~np.isnan(data), np.log1p(data), data)
            data_list.append(data)

            length += adata.shape[0]
            len_list.append(length)
        data_array_1 = np.array(data_list)
        len_array_1 = np.array(len_list)

        return data_array, len_array, data_array_1, len_array_1, length

    def __getitem__(self, index):
        data = self.data_array[0][index]
        data = torch.tensor(data.todense(), dtype=torch.float32)
        # print(data.shape)
        data = data.squeeze()
        # print(data.shape)

        data_1 = self.data_array_1[0][index]
        data_1 = data_1
        mask_idx = torch.tensor(~np.isnan(data_1))
        mask_idx = mask_idx.squeeze()
        data_1 = torch.tensor(np.nan_to_num(data_1, nan=1e-8), dtype=torch.float32)
        data_1 = data_1.squeeze()

        return data_1, mask_idx, 1


# class scRNADataset(Dataset):
#     def __init__(self, data_dir, gene_order_path):
#         self.gene_order_path = gene_order_path
#         self.data_dir = data_dir
#         self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
#         gene_order = pd.read_csv(
#             self.gene_order_path, index_col=0
#         )
#         self.gene_order = gene_order["0"]
#         self.data_array, self.len_array, self.length = self.load_all_data()


#     def __len__(self):
#         return self.length

#     def load_all_data(self):
#         data_list = []
#         len_list = []
#         length = 0

#         for path in self.files:
#             adata = scanpy.read(path, backed="r", cache=True)
#             # order and select gene
#             data_list.append(path)

#             length += adata.shape[0]
#             len_list.append(length)
#         data_array = np.array(data_list)
#         len_array = np.array(len_list)

#         return data_array, len_array, length

#     def __getitem__(self, index):
#         for i in range(len(self.len_array)):
#             if index < self.len_array[i]:
#                 adata = scanpy.read(self.data_array[i], backed="r", cache=True)
#                 file_index = index if i == 0 else index - self.len_array[i - 1]
#                 break
#         adata.var_names = adata.var.feature_id
#         adata = adata[file_index, self.gene_order]
#         data = max_min_normalization(adata.X)
#         data = torch.tensor(data.todense(), dtype=torch.float32)
#         data = data.squeeze()

#         return data


# def dataloader_generator(RNA_path, protein_path, gene_order_path, batch_size=64, num_workers=0):
#     RNA_dataset = scRNADataset(data_dir=RNA_path, gene_order_path=gene_order_path)
#     protein_dataset = scADTDataset(data_dir=protein_path)
#     data_loader = DataLoader(
#         dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
#     )

#     return data_loader


def clr(adata: AnnData, inplace: bool = True, axis: int = 0) -> Union[None, AnnData]:
    """
    Apply the centered log ratio (CLR) transformation
    to normalize counts in adata.X.

    Args:
        data: AnnData object with protein expression counts.
        inplace: Whether to update adata.X inplace.
        axis: Axis across which CLR is performed.
    """

    if axis not in [0, 1]:
        raise ValueError("Invalid value for `axis` provided. Admissible options are `0` and `1`.")

    if not inplace:
        adata = adata.copy()

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warn("adata.X is sparse but not in CSC format. Converting to CSC.")
        x = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warn("adata.X is sparse but not in CSR format. Converting to CSR.")
        x = csr_matrix(adata.X)
    else:
        x = adata.X

    if issparse(x):
        print(list(np.log1p(x).sum(axis=axis).A))
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]), x.getnnz(axis=axis)
        )
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis]),
            out=x,
        )
    print(x)

    adata.X = x

    return None if inplace else adata


def hierarchical_bayesian_downsampling_csr(X_csr, threshold=1000):
    # X_csr is a scipy CSR matrix representing the gene expression matrix
    n, p = X_csr.shape

    # First hierarchy: For each cell, decide whether to downsample based on total expression
    total_expression = np.array(X_csr.sum(axis=1)).flatten()
    gamma = bernoulli.rvs(0.5, size=n)
    gamma[total_expression < threshold] = 0  # Cells below the threshold are not downsampled

    # Second hierarchy: Generate downsampling rates for each cell
    b = beta.rvs(2, 2, size=n)

    # Apply downsampling
    # Since we can't sample directly in the sparse domain, we multiply the non-zero elements
    downsampled_data = X_csr.multiply(gamma[:, np.newaxis]).multiply(b[:, np.newaxis])

    return downsampled_data


def hierarchical_bayesian_downsampling_csr(X_csr, threshold=1000):
    # X_csr is a scipy CSR matrix representing the gene expression matrix
    n, p = X_csr.shape

    # First hierarchy: For each cell, decide whether to downsample based on total expression
    total_expression = np.array(X_csr.sum(axis=1)).flatten()
    gamma = bernoulli.rvs(0.5, size=n)
    gamma[total_expression < threshold] = 0  # Cells below the threshold are not downsampled

    # Second hierarchy: Generate downsampling rates for each cell
    b = beta.rvs(2, 2, size=n)

    # Apply downsampling
    # Apply gamma and b only to non-zero elements of the CSR matrix
    non_zero_mask = X_csr.data > 0  # Identify non-zero elements in the data
    X_csr.data[non_zero_mask] *= gamma[X_csr.indices[non_zero_mask]]  # Apply gamma
    X_csr.data[non_zero_mask] *= b[X_csr.indices[non_zero_mask]]  # Apply b

    return X_csr


def list_files_in_dir(dir_path):
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def max_min_normalization(data):
    High = 1e6
    Low = 1e-8
    scale = High - Low

    min_vals = data.min(axis=1).toarray().ravel()
    max_vals = data.max(axis=1).toarray().ravel()

    nonzero_rows, nonzero_cols = data.nonzero()

    data.data = data.data.astype(float)
    data.data /= max_vals[nonzero_rows] - min_vals[nonzero_rows]
    data.data *= scale
    data.data += Low

    return data


def max_min_normalization_p(data):
    max_value = np.max(data)
    min_value = np.min(data)

    # 进行最大-最小归一化
    normalized_matrix = (data - min_value) / (max_value - min_value)

    return normalized_matrix


def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    x = low + (x - MIN) / (MAX - MIN) * (high - low)  # zoom to (low, high)
    return x


def fetch_files_in_subdirs(dir_path, suffix='.scb'):
    subdirs = []
    for _ in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, _)
        if os.path.isdir(cur_path) and cur_path.endswith(suffix):
            subdirs.append(cur_path)

    file_paths = []
    for subdir in subdirs:
        file_paths.extend(list_files_in_dir(subdir))
    return file_paths


def sort(data):
    sorted_indices = np.argsort(-data, axis=1)
    sorted_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        row_indices = sorted_indices[i]
        sorted_data[i, :] = data[i, row_indices]
    return sorted_data, sorted_indices


class paProteinDataset(Dataset):
    def __init__(self, data_dir, mask_dir='./10M_mask/',
                 reverse_vocab_path='./tokenizer/reverse_vocab.json',
                 origin_vocab_path='./tokenizer/vocab.json',
                 protein_count=6428):
        # print('here')
        # self.gene_order_path = gene_order_path
        self.data_dir = data_dir
        self.protein_count = protein_count
        self.reverse_vocab_path = reverse_vocab_path
        self.origin_vocab_path = origin_vocab_path
        # import json
        # with open('./process.json', 'r') as f:
        #     a = json.load(f)
        # self.dicts = a
        self.files = [sd for sd in fetch_files_in_subdirs(data_dir) if sd.endswith(".parquet")]
        self.files.sort(reverse=False)
        self.masks = []
        for fs in self.files:
            last = fs.split('/')[-2]
            num = last.split('_')[1]
            self.masks.append(mask_dir + 'mask_' + str(num) + '_no_zero.json')
        # from pathlib import Path
        # input_dir = Path(mask_dir)
        # self.masks = [f for f in input_dir.glob("*.json")]
        # self.masks = [f for f in list_files_in_dir(mask_dir) if f.endswith(".json")]
        # self.masks = sorted(self.masks, key=extract_number)
        # self.masks.sort(reverse=False)
        # self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
        # gene_order = pd.read_csv(self.gene_order_path, index_col=0)
        # self.gene_order = gene_order["0"]
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()
        self.fullmasks = self.load_mask_data()
        # self.data_arrayMASK, self.len_arrayMASK, self.lengthMASK = self.load_mask_data()

        # self.fillers = []
        # for i in range(6794):
        #     if str(i) in self.dicts.keys() and self.dicts[str(i)]['avg'] >= 0:
        #         self.fillers.append(self.dicts[str(i)]['avg'])
        #     else:
        #         self.fillers.append(0.0)
        # self.fillers = normalization(torch.tensor(self.fillers))
        # print(list(self.fillers))

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        len_list = []
        length = 0

        for path in self.files:
            # print(path)
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def load_mask_data(self):
        data_list = []

        # for path in self.files:
        for path in self.masks:
            # print(path)
            with open(path, "r") as f:
                mask = json.load(f)
            # adata = scanpy.read(path)
            # adata.var_names = adata.var.feature_id
            # s = "CD57 CD45 CD19 CD45RA CD4 CD8 pSTAT5 CD16 CD127 CD1c CD123 CD66b pSTAT1 CD27 pp38 pSTAT3 pMAPKAP2 CD14 CD56 pPLCg2 CD25 pERK1_2 CD3 CD38 CD161 pS6"
            # s = "CD1a CD1c CD2 CD4 CD5 CD7 CD10 CD11b CD11c CD13 CD14 CD16 CD19 CD20 CD22 CD25 CD26 CD33 CD34 CD36 CD38 CD41 CD42b CD45 CD45RA CD45RO CD48 CD61 CD64 CD66b CD69 CD71 CD99 CD123 CD161 CD163 CD177 IgD IgM CD43 CD140B CD49d CD49f CD54 CD31 CD80 CD86 CD47 CD40 CD154 CD52 CD105 CD44 Podoplanin EGFR CD146 CD32 CD27 CD39 CX3CR1 CD21 CD11a CD235ab CD106 FceRIalpha CD83 CD59 CD29 CD49b CD98 CD55 CD18 CD28 CD204 CD63 CD72 MERTK CD93 CD49a CD9 CD110 CD109 GP130 CD164 CD142 CD45RB CLEC12A CD46 CD94 IgE CD162 CD84 CD23 GPR56 CD82 NKp80 CD131 CD74 CD116 CD37 CD321 CD30 XCR1 CD62E CCR10 CD24 CD70 CD133 C5L2 CD62L CD1d CD35"
            # pros10 = s.split()
            # adata = adata[:, pd.Series(pros10)]
            # adata = adata[:, self.gene_order]
            # adata = adata[:, :]
            # data = max_min_normalization(adata.X)
            # order and select gene
            dataMatrix = np.array(list(mask.values()), dtype=object)
            data_list.append(dataMatrix)

        data_arrayMASK = np.array(data_list, dtype=object)

        return data_arrayMASK

    def __getitem__(self, index):
        self.vocab = self.get_reverse_vocab()
        self.gene_vocab = self.get_origin_vocab()

        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                fullmask = self.fullmasks[i]
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        # 根据随机index计算出当前row group号与group中具体第几行
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])
        # 对一个row group进行列索引，获得对应信息（as_py()方法用于将数据类型转化为python中的基本数据类型）
        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()
        # 新建一个data向量用于存储表达矩阵
        data = np.zeros(self.protein_count)

        # 直接通过genes数组作为索引数组索引data，对应位置就是有表达量的位置
        data[pgenes] = pexps
        # assert len(pexps) == len(fullmask[str(file_index)])
        # data[fullmask[str(file_index)]] = pexps
        # nan2zero
        data = np.nan_to_num(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()
        tech = data[-1]
        data = data[:-1]

        # if tech.item() == 0.0:
        #     data = cytof(data)
        # elif tech.item() == 1.0:
        #     data = citeseq(np.array(data))
        # print(tech)
        # print(data)
        # print(data.shape) 6794

        mask_position = np.array(fullmask[file_index][:-1], dtype=int)
        mask = np.zeros(len(data))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)
        # print(indices)
        # for pos in indices:
        #     # print(pos)
        #     if str(pos.item()) in self.dicts.keys():
        #         data[pos.item()] = self.dicts[str(pos.item())]['avg']
        #         # print(self.dicts[str(pos.item())]['avg'])
        #     # else:
        #     #     data[pos.item()] = 0.0
        # for pos in range(data.shape[0]):
        #     # undetected known
        #     if pos not in indices and str(pos) in self.dicts.keys():
        #         data[pos] = self.dicts[str(pos)]['avg'] + torch.rand(1).item()
        #         # print(data[pos])
        #     elif pos not in indices and str(pos) not in self.dicts.keys():
        #         data[pos] = torch.rand(1).item()

        # data = normalization(data)
        # data = np.log1p(data)
        # for pos in range(data.shape[0]):
        #     # undetected known
        #     if pos not in indices and str(pos) in self.dicts.keys():
        #         # data[pos] = self.dicts[str(pos)]['avg'] + torch.rand(1).item()
        #         data[pos] = 0.0
        #         # print(data[pos])
        #     elif pos not in indices and str(pos) not in self.dicts.keys():
        #         data[pos] = 0.0
        #         # data[pos] = torch.rand(1).item() * 5

        if torch.isnan(data).any():
            print("hahaha")
            print(list(data))
        data = normalization(data)
        # data = np.log1p(data)
        if torch.isnan(data).any():
            print("hahaha")
            print(list(data))
        # for pos in range(data.shape[0]):
        #     if pos not in indices:
        #         data[pos] = 0.0
        # print(data)
        return (data, mask, tech)


class paTESTProteinDataset_cytof(Dataset):
    def __init__(self, data_dir,
                 reverse_vocab_path='./tokenizer/reverse_vocab.json',
                 origin_vocab_path='./tokenizer/vocab.json',
                 mask_test_path='./mask_test_cytof.npy', protein_count=6428):
        # print('here')
        # self.gene_order_path = gene_order_path
        self.data_dir = data_dir
        self.protein_count = protein_count
        self.reverse_vocab_path = reverse_vocab_path
        self.origin_vocab_path = origin_vocab_path
        # import json
        # with open('./process.json', 'r') as f:
        #     a = json.load(f)
        # self.dicts = a
        self.files = [sd for sd in fetch_files_in_subdirs(data_dir) if sd.endswith(".parquet")]
        self.files.sort(reverse=False)
        # self.masks = [f for f in list_files_in_dir(data_dir) if f.endswith(".json")]
        # self.masks.sort(reverse=True)
        self.mask = np.load(mask_test_path)
        self.mask = torch.tensor(self.mask, dtype=torch.int)
        self.mask = self.mask.squeeze()
        # self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
        # gene_order = pd.read_csv(self.gene_order_path, index_col=0)
        # self.gene_order = gene_order["0"]
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()
        # self.fullmasks = self.load_mask_data()
        # self.data_arrayMASK, self.len_arrayMASK, self.lengthMASK = self.load_mask_data()

        # self.reverse_vocab_path = reverse_vocab_path
        # self.origin_vocab_path = origin_vocab_path
        # self.fillers = []
        # for i in range(6794):
        #     if str(i) in self.dicts.keys() and self.dicts[str(i)]['avg'] >= 0:
        #         self.fillers.append(self.dicts[str(i)]['avg'])
        #     else:
        #         self.fillers.append(0.0)
        # self.fillers = normalization(torch.tensor(self.fillers))
        # print(list(self.fillers))

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        self.vocab = self.get_reverse_vocab()
        self.gene_vocab = self.get_origin_vocab()
        len_list = []
        length = 0

        for path in self.files:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def __getitem__(self, index):
        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                # fullmask = self.fullmasks[i]
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        # 根据随机index计算出当前row group号与group中具体第几行
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])
        # 对一个row group进行列索引，获得对应信息（as_py()方法用于将数据类型转化为python中的基本数据类型）
        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()
        # 新建一个data向量用于存储表达矩阵
        data = np.zeros(self.protein_count)
        # 直接通过genes数组作为索引数组索引data，对应位置就是有表达量的位置
        data[pgenes] = pexps
        # nan2zero
        data = np.nan_to_num(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()
        tech = data[-1]
        data = data[:-1]

        # data = cytof(data)
        # print(tech)
        # print(data)
        # print(data.shape) 6794

        mask_position = np.array(self.mask[:-1])
        mask = np.zeros(len(data))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)

        if torch.isnan(data).any():
            print("hahaha")
            print(list(data))
        data = normalization(data)
        # data = np.log1p(data)
        # for pos in range(data.shape[0]):
        #     if pos not in indices:
        #         data[pos] = 0.0
        # print(data)
        return (data, mask, tech)


class paTESTProteinDataset_citeseq(Dataset):
    def __init__(self, data_dir,
                 reverse_vocab_path='./tokenizer/reverse_vocab.json',
                 origin_vocab_path='./tokenizer/vocab.json',
                 mask_test_path='./mask_test_citeseq.npy', protein_count=6428):
        # print('here')
        # self.gene_order_path = gene_order_path
        self.data_dir = data_dir
        self.protein_count = protein_count
        self.reverse_vocab_path = reverse_vocab_path
        self.origin_vocab_path = origin_vocab_path
        # import json
        # with open('./process.json', 'r') as f:
        #     a = json.load(f)
        # self.dicts = a
        self.files = [sd for sd in fetch_files_in_subdirs(data_dir) if sd.endswith(".parquet")]
        self.files.sort(reverse=False)
        # self.masks = [f for f in list_files_in_dir(data_dir) if f.endswith(".json")]
        # self.masks.sort(reverse=True)
        self.mask = np.load(mask_test_path)
        self.mask = torch.tensor(self.mask, dtype=torch.int)
        self.mask = self.mask.squeeze()
        # self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
        # gene_order = pd.read_csv(self.gene_order_path, index_col=0)
        # self.gene_order = gene_order["0"]
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()
        # self.fullmasks = self.load_mask_data()
        # self.data_arrayMASK, self.len_arrayMASK, self.lengthMASK = self.load_mask_data()

        # self.reverse_vocab_path = reverse_vocab_path
        # self.origin_vocab_path = origin_vocab_path
        # self.fillers = []
        # for i in range(6794):
        #     if str(i) in self.dicts.keys() and self.dicts[str(i)]['avg'] >= 0:
        #         self.fillers.append(self.dicts[str(i)]['avg'])
        #     else:
        #         self.fillers.append(0.0)
        # self.fillers = normalization(torch.tensor(self.fillers))
        # print(list(self.fillers))

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        self.vocab = self.get_reverse_vocab()
        self.gene_vocab = self.get_origin_vocab()
        len_list = []
        length = 0

        for path in self.files:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def __getitem__(self, index):
        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                # fullmask = self.fullmasks[i]
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        # 根据随机index计算出当前row group号与group中具体第几行
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])
        # 对一个row group进行列索引，获得对应信息（as_py()方法用于将数据类型转化为python中的基本数据类型）
        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()
        # 新建一个data向量用于存储表达矩阵
        data = np.zeros(self.protein_count)
        # 直接通过genes数组作为索引数组索引data，对应位置就是有表达量的位置
        data[pgenes] = pexps
        # nan2zero
        data = np.nan_to_num(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()
        tech = data[-1]
        data = data[:-1]

        # data = citeseq(np.array(data))
        # print(tech)
        # print(data)
        # print(data.shape) 6794

        mask_position = np.array(self.mask[:-1])
        mask = np.zeros(len(data))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)

        if torch.isnan(data).any():
            print("hahaha")
            print(list(data))
        data = normalization(data)
        # data = np.log1p(data)
        # for pos in range(data.shape[0]):
        #     if pos not in indices:
        #         data[pos] = 0.0
        # print(data)
        return (data, mask, tech)


def dataloader_generator(root_path, gene_order_path, batch_size=64, num_workers=0):
    dataset = paProteinDataset(data_dir=root_path)
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return data_loader