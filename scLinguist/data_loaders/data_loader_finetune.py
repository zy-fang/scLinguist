import json
import os
import random
import scanpy
import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix, issparse, csc_matrix
from warnings import warn
from typing import Union
from anndata import AnnData
import scipy.sparse as sp
import pyarrow.parquet as pq
from scipy.stats import bernoulli, beta


def hierarchical_bayesian_downsampling_csr(X_csr, threshold=1000):
    # X_csr is a scipy CSR matrix representing the gene expression matrix
    n, p = X_csr.shape

    # First hierarchy: For each cell, decide whether to downsample based on total expression
    total_expression = np.array(X_csr.sum(axis=1)).flatten()
    gamma = bernoulli.rvs(0.5, size=n)
    gamma[
        total_expression < threshold
        ] = 0  # Cells below the threshold are not downsampled

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
    gamma[
        total_expression < threshold
        ] = 0  # Cells below the threshold are not downsampled

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
    High = 1
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


def sort(data):
    sorted_indices = np.argsort(-data, axis=1)
    sorted_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        row_indices = sorted_indices[i]
        sorted_data[i, :] = data[i, row_indices]
    return sorted_data, sorted_indices

def max_min_normalization_with_nan(data):
    High = 1000000
    Low = 1
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
            data[i] /= max_vals[i] - min_vals[i]
        data[i] *= scale
        data[i] += Low

    return data


def clr(adata, inplace=True, axis=0):
    if axis not in [0, 1]:
        raise ValueError(
            "Invalid value for `axis` provided. Admissible options are `0` and `1`."
        )

    if not inplace:
        adata = adata.copy()

    x = adata.X
    if issparse(x):
        nan_mask = np.isnan(x.data)
        x.data[nan_mask] = 0  # Temporarily replace nan with 0
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]), x.getnnz(axis=axis)
        )
        x.data = np.log1p(x.data)
        x.data[nan_mask] = np.nan  # Replace back the nan values
    else:
        nan_mask = np.isnan(x)
        x = np.where(nan_mask, 0, x)  # Temporarily replace nan with 0
        x = np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis])
        )
        x[nan_mask] = np.nan  # Replace back the nan values

    adata.X = x

    return None if inplace else adata


class scRNADataset(Dataset):
    def __init__(self, data_dir, gene_order_path):
        self.gene_order_path = gene_order_path
        self.data_dir = data_dir
        # self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
        gene_order = pd.read_csv(self.gene_order_path, index_col=0)
        self.gene_order = gene_order["0"]
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()

    def __len__(self):
        return self.length

    def load_all_data(self):
        data_list = []
        sorted_data_list = []
        feature_id_list = []
        len_list = []
        length = 0

        # for path in self.files:
        for path in [self.data_dir]:
            adata = scanpy.read(path)
            adata.var_names = adata.var.feature_id
            adata = adata[:, self.gene_order]
            # tmp = hierarchical_bayesian_downsampling_csr(adata.X)
            # adata = anndata.AnnData(tmp)

            scanpy.pp.normalize_total(adata, target_sum=10000)
            scanpy.pp.log1p(adata)
            data = adata.X
            # data = max_min_normalization(adata.X)
            # sorted_data, feature_id = sort(data.todense())
            # order and select gene
            data_list.append(data)
            # sorted_data_list.append(sorted_data)
            # feature_id_list.append(feature_id)

            length += adata.shape[0]
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        # sorted_data_array = np.array(sorted_data_list)
        # feature_id_array = np.array(feature_id_list)

        return data_array, len_array, length

    def __getitem__(self, index):
        data = self.data_array[0][index]
        data = torch.tensor(data.todense(), dtype=torch.float32)
        data = data.squeeze()
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
        for path in [self.data_dir_1]:
            adata = scanpy.read(path)
            scanpy.pp.normalize_total(adata, target_sum=10000)
            scanpy.pp.log1p(adata)
            data = adata.X
            data_list.append(data)

            length += adata.shape[0]
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)

        data_list = []
        len_list = []
        length = 0

        # for path in self.files:
        for path in [self.data_dir_2]:
            adata = scanpy.read(path)
            data = adata.X.todense()

            data = max_min_normalization_with_nan(data)
            data = np.where(~np.isnan(data), np.log1p(data), data)
            # clr(adata)
            # data = adata.X
            data_list.append(data)

            length += adata.shape[0]
            len_list.append(length)
        data_array_1 = np.array(data_list)
        len_array_1 = np.array(len_list)

        return data_array, len_array, data_array_1, len_array_1, length

    def __getitem__(self, index):
        data = self.data_array[0][index]
        data = torch.tensor(data.todense(), dtype=torch.float32)
        data = data.squeeze()

        data_1 = self.data_array_1[0][index]
        data_1 = data_1
        mask_idx = torch.tensor(~np.isnan(data_1))
        mask_idx = mask_idx.squeeze()
        data_1 = torch.tensor(np.nan_to_num(data_1), dtype=torch.float32)
        data_1 = data_1.squeeze()

        return data, data_1, mask_idx


def normalization(x, low=1, high=1e6):
    MIN = min(x)
    MAX = max(x)
    if MIN == MAX:
        x = x
    else:
        x = low + (x - MIN) / (MAX - MIN) * (high - low)  # zoom to (low, high)
    return x


def citeseq(ts):
    x = torch.from_numpy(ts)

    # 计算 torch.log1p(x)
    log1p_x = torch.log1p(x)

    # 计算 torch.log1p(x).sum() / x.shape[0]
    sum_log1p_x = torch.sum(log1p_x) / x.shape[0]

    # 计算 torch.exp(torch.log1p(x).sum() / x.shape[0])
    exp_sum_log1p_x = torch.exp(sum_log1p_x)

    # 计算 x / torch.exp(torch.log1p(x).sum() / x.shape[0])
    result = x / exp_sum_log1p_x

    # 计算 torch.log1p(x / torch.exp(torch.log1p(x).sum() / x.shape[0]))
    final_result = torch.log1p(result)

    return final_result


def find_parquet_files(root_folder, suffix=".parquet"):
    parquet_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(suffix):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


class spMultiDataset(Dataset):
    def __init__(self, data_dir_1, data_dir_2, mask_dir):
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.files_1 = find_parquet_files(data_dir_1)
        print(self.files_1)
        self.files_1.sort(reverse=True)

        self.files_2 = find_parquet_files(data_dir_2)
        print(self.files_2)
        self.files_2.sort(reverse=True)
        self.masks = []
        for fs in self.files_2:
            print(fs)
            last = fs.split("/")[-1]
            num = last.split("_")[1].split(".")[0]
            self.masks.append(mask_dir + "ADT_" + str(num) + ".json")

        (
            self.data_array,
            self.len_array,
            self.data_array_1,
            self.len_array_1,
            self.length,
        ) = self.load_all_data()
        self.fullmasks = self.load_mask_data()

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

        for path in self.files_1:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)

        data_list = []
        len_list = []
        length = 0

        for path in self.files_2:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array_1 = np.array(data_list)
        len_array_1 = np.array(len_list)

        return data_array, len_array, data_array_1, len_array_1, length

    def load_mask_data(self):
        data_list = []

        for path in self.masks:
            with open(path, "r") as f:
                mask = json.load(f)
            dataMatrix = np.array(list(mask.values()), dtype=object)
            data_list.append(dataMatrix)

        data_arrayMASK = np.array(data_list, dtype=object)
        return data_arrayMASK

    def parquet_getitem(self, index, len_array, data_array, data_count):
        for i in range(len(len_array)):
            if index < len_array[i]:
                pdata = pq.ParquetFile(data_array[i])
                file_index = index if i == 0 else index - len_array[i - 1]
                break
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])

        pgenes = pdata["genes"][rows].as_py()
        pexps = pdata["expressions"][rows].as_py()

        data = np.zeros(data_count)
        data[pgenes] = pexps
        return data, file_index, i

    def __getitem__(self, index):
        data, _, _ = self.parquet_getitem(index, self.len_array, self.data_array, 19202)
        data = data.reshape(1, -1)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()

        data_1, file_index, i = self.parquet_getitem(index, self.len_array_1, self.data_array_1, 6427)
        data_1 = torch.tensor(data_1, dtype=torch.float32)
        data_1 = data_1.squeeze()

        fullmask = self.fullmasks[i]
        mask_position = np.array(fullmask[int(file_index)][:-1], dtype=int)
        mask = np.zeros(len(data_1))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)

        data_1 = normalization(data_1, low=1e-8, high=1)


        return data, data_1, mask
