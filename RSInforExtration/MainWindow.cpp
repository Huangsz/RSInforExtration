#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif

//gdal

#include <ogrsf_frmts.h>
#include <gdal_alg.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <gdalwarper.h> 
#include <cpl_conv.h> // for CPLMalloc()
#include <array>
#include <iostream>
#include "RSInforExtraction.h"
#include "GdalOpencvTensor.h"
#include "projection.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>



int main()
{
	
    //路径设置
    std::string input_image_tif =       "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1.tif";  //输入影像路径
    std::string input_image_tif_float = "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_float.tif";//转换输入tif影像为float tif路径
    std::string Result_shpfile_Name =   "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result.shp";//输出shp文件路径
    std::string Result_tiffile_Name =   "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result.tif";//输出tif文件路径
    std::string Shpfile_trans_Name =    "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result_trans.shp";//转换后的shp文件路径
    std::string Shpfile_simplify =      "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1result_simplify.shp";//简化后的shp文件路径
    std::string DLL_dir =               "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\Deep_learning\\dll";//深度学习模型路径
    std::string input_img_for_show =    "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1.jpg";//tif转换为jpg用于展示原始影像的路径
    std::string output_img_for_show =   "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result_show.jpg";//    tif转换为jpg用于展示结果的路径
    std::string dsmFilePath =           "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E114.0_N22.5_Nanshan_Mosaic_DSM_2m _test1.tif";//   DSM文件路径
    std::string outputDsmFilePath =     "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E114.0_N22.5_Nanshan_Mosaic_DSM_2m _test1_trans.tif";// 转换后的DSM文件路径
    std::string OBJ_FilePath =          "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result.obj";//输出的OBJ文件路径    
    std::string projLibPath =           "D:\\project\\GDAL\\gdal-3.8.5\\bin\\proj9\\share";   // proj库路径，即proj.db文件所在路径

    if (std::getenv("PROJ_LIB") == nullptr) {
        std::cout << "Setting PROJ_LIB environment variable to " << projLibPath << std::endl;
        _putenv_s("PROJ_LIB", projLibPath.c_str()); // 在 Windows 上设置环境变量
    }
    GDALAllRegister();  //注册驱动
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");   //支持中文路径
	GDALDataset *m_poDataSet = (GDALDataset*)GDALOpen(input_image_tif.c_str(), GA_ReadOnly);    //GDAL数据集
	if (!m_poDataSet){
		cout << "fail in open files!!!" << endl;
		return 0;
	}
	int nCols = m_poDataSet->GetRasterXSize();
	int nRows = m_poDataSet->GetRasterYSize();
	int nBands = m_poDataSet->GetRasterCount();
	GDALDriver* poDriver_new = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (poDriver_new == NULL) {
		std::cerr << "GTiff driver not available." << std::endl;
		GDALClose(m_poDataSet);
	}
	//获取影像基本信息
	//获取仿射矩阵(坐标变换系数)，含有6个元素的元组
	double adfGeoTransform[6];
	m_poDataSet->GetGeoTransform(adfGeoTransform);
	//影像读取为Mat格式  不分块
	GdalOpencvTensor gdalOpenCV;
	cv::Mat imagedata_input_for_net;    //开辟空间存储影像数据 
	gdalOpenCV.GDAL2Mat(input_image_tif, imagedata_input_for_net); //GDAL Type ==========> GDALOpenCV Type 
	gdalOpenCV.TIF2JPG(input_image_tif, input_image_tif_float, input_img_for_show);

    tensorflow::Tensor input_img_tensor = gdalOpenCV.Mat2Tensor(imagedata_input_for_net);     //将opencv的数据转换为tensor张量数据
    RSInforExtraction* Rs = new RSInforExtraction();
    tensorflow::uint8* result_tensor = Rs->tensorflow_building(input_img_tensor, nCols, DLL_dir);
    GDALDataset* poDataset2 = gdalOpenCV.save_result_tif(m_poDataSet, Result_tiffile_Name, adfGeoTransform, result_tensor);
    gdalOpenCV.tif2shp(poDataset2, m_poDataSet,Result_shpfile_Name);
    delete Rs;
    GDALClose(poDataset2);
    GDALClose(m_poDataSet);
    gdalOpenCV.Result_TIF2JPG(Result_tiffile_Name, output_img_for_show);

	double areaThreshold = 100.0;
	double simplifyTolerance = 0.9;
	std::vector<OGRPolygon*> polygons;
	OGRSpatialReference* sourceSRS = nullptr;
    OGRSpatialReference* targetSRS = nullptr;
    gdalOpenCV.readShapefile(Result_shpfile_Name.c_str(), polygons, &sourceSRS);
    gdalOpenCV.transformToUTM(polygons, sourceSRS, targetSRS);
    gdalOpenCV.writePolygonShapefile(Shpfile_trans_Name.c_str(), polygons, areaThreshold, simplifyTolerance, targetSRS);

    // 转换 DSM 文件
    gdalOpenCV.convertDsmToShpSrs(dsmFilePath, Shpfile_trans_Name, outputDsmFilePath);

    // 提取建筑物轮廓并计算高度范围
    auto buildingFootprints = gdalOpenCV.extractBuildingFootprints(Shpfile_trans_Name, outputDsmFilePath);

    projection::Mesh mesh;
    // 生成LOD1模型
    for (const auto& building : buildingFootprints) {

        //double height = building.second.second - building.second.first; // 最高点减去最低点得到建筑物高度
        projection::buildingLod1Mesh lod1Mesh(building.first, building.second.first, building.second.second);
        lod1Mesh.buildMesh(mesh);
    }
    // 保存 mesh 到 OBJ 文件
    std::ofstream out(OBJ_FilePath);
    if (out.is_open()) {
        CGAL::IO::write_OBJ(out, mesh);
        std::cout << "Mesh saved to output.obj" << std::endl;
    }
    else {
        std::cerr << "Failed to open output.obj for writing" << std::endl;
    }

	return EXIT_SUCCESS;
}