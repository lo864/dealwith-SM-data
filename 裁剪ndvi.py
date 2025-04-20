# 导入 xarray 库，用于处理多维数组和数据集
import xarray as xr
# 导入 geopandas 库，用于处理地理空间数据
import geopandas as gpd
# 导入 rioxarray 库，为 xarray 提供地理空间处理功能
import rioxarray
# 导入 numpy 库，用于科学计算
import numpy as np
# 导入 os 库，用于与操作系统进行交互，如文件和目录操作
import os
# 从 pathlib 模块导入 Path 类，用于更方便地处理文件路径
from pathlib import Path

# 设置路径
# 原始数据目录，存放每日 NDVI 数据的 NetCDF 文件
raw_data_dir = "D:/BaiduNetdiskDownload/data/ndvi"
# 输出目录，用于保存处理后的结果
output_dir = "D:/BaiduNetdiskDownload/figures"
# 流域 shapefile 文件的路径
basin_shp_path = "E:/iCloudDrive/Desktop/博士期间/毕业论文/data/30子流域.shp"
# 合并后每年 NDVI 数据的 NetCDF 文件路径
nc_yearly_path = os.path.join(output_dir, "ndvi_yearly_1986_2022.nc")  # 更新文件名

# 确保输出目录存在
# 如果目录不存在，则创建该目录；如果已存在，则不会报错
os.makedirs(output_dir, exist_ok=True)

# 读取流域 shapefile 并转换为 WGS84 坐标系
# 使用 geopandas 读取 shapefile 文件
basin_shp = gpd.read_file(basin_shp_path).to_crs("EPSG:4326")

# 获取流域边界框并添加缓冲区
# 获取流域的边界框，格式为 [最小经度, 最小纬度, 最大经度, 最大纬度]
bounds = basin_shp.total_bounds  # [minx, miny, maxx, maxy]
# 缓冲区大小
buffer = 0.1
# 计算添加缓冲区后的边界框
minx, miny, maxx, maxy = bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer
# 打印添加缓冲区后的边界框
print("Bounds after buffer:", minx, miny, maxx, maxy)

# 收集所有年份的 NDVI 数据
# 用于存储每年处理后的 NDVI 数据的列表
ndvi_yearly_list = []
# 遍历 1985 到 2022 年（调整为处理 1986 - 2022 年的数据）
for year in range(1985, 2023):  # 调整为 1986-2022
    # 构建每年原始 NDVI 数据文件的路径
    raw_file = os.path.join(raw_data_dir, f"Daily_Gap-filled_NDVI_{year}.nc4")
    # 检查文件是否存在
    if not os.path.exists(raw_file):
        # 如果文件不存在，打印警告信息
        print(f"Warning: File for year {year} not found: {raw_file}")
        # 跳过当前年份，继续处理下一年
        continue

    # 加载每日 NDVI 数据
    # 使用 xarray 打开 NetCDF 文件，并设置分块大小
    ds = xr.open_dataset(raw_file, chunks={'time': 100, 'lat': 100, 'lon': 100})
    
    # 调试：打印数据集结构
    # 打印当前年份的数据集信息
    print(f"\nDataset for year {year}:")
    print(ds)
    # 打印数据集中的变量名
    print("Variables:", list(ds.variables))
    # 打印数据集的维度名
    print("Dimensions:", list(ds.dims))
    # 打印 NDVI 变量的维度名
    print("NDVI dimensions:", ds['NDVI'].dims)

    # 检查 NDVI 变量是否存在
    if 'NDVI' not in ds:
        # 如果 NDVI 变量不存在，打印错误信息
        print(f"Error: 'NDVI' variable not found in {raw_file}")
        # 跳过当前年份，继续处理下一年
        continue

    # 获取 NDVI 变量
    ndvi = ds['NDVI']

    # 确保维度名正确
    # 期望的维度名集合
    expected_dims = {'lat', 'lon', 'time'}
    # 实际的 NDVI 维度名集合
    actual_dims = set(ndvi.dims)
    # 检查期望的维度名是否都在实际维度名中
    if not expected_dims.issubset(actual_dims):
        # 如果不满足，打印错误信息
        print(f"Error: Expected dimensions {expected_dims} not found in NDVI. Actual dimensions: {actual_dims}")
        # 尝试重命名维度（如果维度名是大写）
        # 用于存储维度重命名的字典
        rename_dict = {}
        # 遍历 NDVI 的维度名
        for dim in ndvi.dims:
            if dim.lower() == 'lat':
                # 如果维度名小写后是 'lat'，则将其重命名为 'lat'
                rename_dict[dim] = 'lat'
            elif dim.lower() == 'lon':
                # 如果维度名小写后是 'lon'，则将其重命名为 'lon'
                rename_dict[dim] = 'lon'
            elif dim.lower() == 'time':
                # 如果维度名小写后是 'time'，则将其重命名为 'time'
                rename_dict[dim] = 'time'
        if rename_dict:
            # 如果有需要重命名的维度，打印重命名信息
            print(f"Renaming dimensions: {rename_dict}")
            # 重命名 NDVI 的维度
            ndvi = ndvi.rename(rename_dict)

    # 设置空间维度和 CRS
    try:
        # 使用 rioxarray 设置 NDVI 的空间维度和坐标系
        ndvi = ndvi.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs("EPSG:4326")
    except Exception as e:
        # 如果设置过程中出现异常，打印错误信息
        print(f"Error setting spatial dims for year {year}: {e}")
        # 手动设置坐标和 CRS
        if 'lon' in ds and 'lat' in ds:
            # 如果数据集中存在 'lon' 和 'lat' 变量，手动为 NDVI 分配坐标
            ndvi = ndvi.assign_coords(lon=ds['lon'], lat=ds['lat'])
            # 手动写入坐标系
            ndvi.rio.write_crs("EPSG:4326", inplace=True)
        else:
            # 如果无法手动设置坐标，打印错误信息并跳过当前年份
            print(f"Error: Cannot set spatial dims for year {year}. Skipping.")
            continue

    # 裁剪到流域边界框
    # 使用边界框对 NDVI 数据进行裁剪
    ndvi = ndvi.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))

    # 使用流域 shapefile 进行掩膜
    # 使用流域的几何形状对 NDVI 数据进行掩膜处理
    ndvi = ndvi.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True).rio.write_crs("EPSG:4326")

    # 计算年均 NDVI
    # 按时间维度计算 NDVI 的平均值，忽略缺失值
    ndvi_yearly = ndvi.mean(dim='time', skipna=True)
    # 为年均 NDVI 数据添加年份维度
    ndvi_yearly = ndvi_yearly.expand_dims({'year': [year]})

    # 将年均 NDVI 数据添加到列表中
    ndvi_yearly_list.append(ndvi_yearly)
    # 打印处理完成的年份信息
    print(f"Processed year: {year}")

# 合并所有年份的数据
# 如果列表为空，说明没有处理到有效的 NDVI 数据，抛出异常
if not ndvi_yearly_list:
    raise ValueError("No NDVI data processed. Please check the raw data files.")
# 沿着年份维度合并所有年份的年均 NDVI 数据
ndvi_yearly = xr.concat(ndvi_yearly_list, dim='year')

# 保存为 NetCDF 文件
# 将合并后的年均 NDVI 数据保存为 NetCDF 文件
ndvi_yearly.to_netcdf(nc_yearly_path)
# 打印保存的文件路径信息
print(f"Yearly NDVI data saved to: {nc_yearly_path}")

# 计算 1986-2022 年的均值并保存
# 按年份维度计算年均 NDVI 的平均值，忽略缺失值
ndvi_mean = ndvi_yearly.mean(dim='year', skipna=True)
# 构建保存均值数据的 NetCDF 文件路径
nc_mean_path = os.path.join(output_dir, "ndvi_mean_1986_2022.nc")
# 将均值数据保存为 NetCDF 文件
ndvi_mean.to_netcdf(nc_mean_path)
# 打印保存的均值数据文件路径信息
print(f"Mean NDVI data saved to: {nc_mean_path}")

# 检查生成的年份范围
# 打开合并后的年均 NDVI 数据文件
ds = xr.open_dataset(nc_yearly_path)
# 打印数据集中的年份值
print("Years in ndvi_yearly:", ds['year'].values)