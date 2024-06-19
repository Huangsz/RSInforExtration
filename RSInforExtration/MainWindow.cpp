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
	
    //·������
    std::string input_image_tif =       "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1.tif";  //����Ӱ��·��
    std::string input_image_tif_float = "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_float.tif";//ת������tifӰ��Ϊfloat tif·��
    std::string Result_shpfile_Name =   "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result.shp";//���shp�ļ�·��
    std::string Result_tiffile_Name =   "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result.tif";//���tif�ļ�·��
    std::string Shpfile_trans_Name =    "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result_trans.shp";//ת�����shp�ļ�·��
    std::string Shpfile_simplify =      "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1result_simplify.shp";//�򻯺��shp�ļ�·��
    std::string DLL_dir =               "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\Deep_learning\\dll";//���ѧϰģ��·��
    std::string input_img_for_show =    "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1.jpg";//tifת��Ϊjpg����չʾԭʼӰ���·��
    std::string output_img_for_show =   "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result_show.jpg";//    tifת��Ϊjpg����չʾ�����·��
    std::string dsmFilePath =           "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E114.0_N22.5_Nanshan_Mosaic_DSM_2m _test1.tif";//   DSM�ļ�·��
    std::string outputDsmFilePath =     "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E114.0_N22.5_Nanshan_Mosaic_DSM_2m _test1_trans.tif";// ת�����DSM�ļ�·��
    std::string OBJ_FilePath =          "D:\\project\\GF\\CODE_LH\\RSInforExtration\\RSInforExtration\\test_data1\\new\\GF7_E113.9_N22.5_20210421_DOM_0.7m_test1_result.obj";//�����OBJ�ļ�·��    
    std::string projLibPath =           "D:\\project\\GDAL\\gdal-3.8.5\\bin\\proj9\\share";   // proj��·������proj.db�ļ�����·��

    if (std::getenv("PROJ_LIB") == nullptr) {
        std::cout << "Setting PROJ_LIB environment variable to " << projLibPath << std::endl;
        _putenv_s("PROJ_LIB", projLibPath.c_str()); // �� Windows �����û�������
    }
    GDALAllRegister();  //ע������
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");   //֧������·��
	GDALDataset *m_poDataSet = (GDALDataset*)GDALOpen(input_image_tif.c_str(), GA_ReadOnly);    //GDAL���ݼ�
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
	//��ȡӰ�������Ϣ
	//��ȡ�������(����任ϵ��)������6��Ԫ�ص�Ԫ��
	double adfGeoTransform[6];
	m_poDataSet->GetGeoTransform(adfGeoTransform);
	//Ӱ���ȡΪMat��ʽ  ���ֿ�
	GdalOpencvTensor gdalOpenCV;
	cv::Mat imagedata_input_for_net;    //���ٿռ�洢Ӱ������ 
	gdalOpenCV.GDAL2Mat(input_image_tif, imagedata_input_for_net); //GDAL Type ==========> GDALOpenCV Type 
	gdalOpenCV.TIF2JPG(input_image_tif, input_image_tif_float, input_img_for_show);

    tensorflow::Tensor input_img_tensor = gdalOpenCV.Mat2Tensor(imagedata_input_for_net);     //��opencv������ת��Ϊtensor��������
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

    // ת�� DSM �ļ�
    gdalOpenCV.convertDsmToShpSrs(dsmFilePath, Shpfile_trans_Name, outputDsmFilePath);

    // ��ȡ����������������߶ȷ�Χ
    auto buildingFootprints = gdalOpenCV.extractBuildingFootprints(Shpfile_trans_Name, outputDsmFilePath);

    projection::Mesh mesh;
    // ����LOD1ģ��
    for (const auto& building : buildingFootprints) {

        //double height = building.second.second - building.second.first; // ��ߵ��ȥ��͵�õ�������߶�
        projection::buildingLod1Mesh lod1Mesh(building.first, building.second.first, building.second.second);
        lod1Mesh.buildMesh(mesh);
    }
    // ���� mesh �� OBJ �ļ�
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