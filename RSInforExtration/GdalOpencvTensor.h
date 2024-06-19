#pragma once
#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif

#include <iostream>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cxcore.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>
#include <gdal_alg.h>
#include <gdal.h>
#include <cpl_conv.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/create_straight_skeleton_2.h>
#include <CGAL/Straight_skeleton_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include "projection.h"


#define COMPILER_MSVC
#define NOMINMAX
#include "tensorflow/core/framework/tensor.h"



typedef enum {
	GC_Byte = 0,
	GC_UInt16 = 1,
	GC_Int16 = 2,
	GC_UInt32 = 3,
	GC_Int32 = 4,
	GC_Float32 = 5,
	GC_Float64 = 6,
	GC_ERRType = 7
} GCDataType;

using namespace std;
typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Polygon_2<Kernel> Polygon_2;
typedef CGAL::Straight_skeleton_2<Kernel> Ss;
typedef boost::shared_ptr<Ss> SsPtr;
typedef CGAL::Polygon_with_holes_2<Kernel> Polygon_with_holes_2;


class GdalOpencvTensor
{
public:
	//struct PolygonData {
	//	OGRPolygon* polygon;
	//	OGREnvelope envelope;
	//};
	GdalOpencvTensor();
	~GdalOpencvTensor();
public:
	
	tensorflow::Tensor Mat2Tensor(cv::Mat& img);

	bool GDAL2Mat(std::string input_image_tif, cv::Mat& img);  // 影像读取为Mat格式  不分块
	bool TIF2JPG(std::string input_image_tif, std::string input_image_tif_float, std::string outputFilename);
	void histogramEqualization(GDALRasterBand* poBand, cv::Mat& outputMat);
	void tif2shp(GDALDataset* poDataset, GDALDataset* poDataset_prj_ref, std::string outFilePath);
    GDALDataset* save_result_tif(GDALDataset* m_poDataSet, const std::string& Result_tiffile_Name, double* adfGeoTransform, tensorflow::uint8* result_tensor);
    void Result_TIF2JPG(std::string Result_tiffile_Name, std::string input_img_for_show);
	/// ////////////////////////////////////////////////////////////////////////////
	double calculatePolygonArea(std::vector<OGRPoint>& polygonData, double* adfGeoTransform);
    void transformToUTM(std::vector<OGRPolygon*>& polygons, OGRSpatialReference* sourceSRS, OGRSpatialReference*& targetSRS);
	int getUTMZone(double longitude);
	bool importSRSFromPrj(const char* prjFilePath, OGRSpatialReference*& sourceSRS);
	void readShapefile(const char* filename, std::vector<OGRPolygon*>& polygons, OGRSpatialReference** sourceSRS);
	void writePolygonShapefile(const char* outputShapefile, std::vector<OGRPolygon*>& polygons, double areaThreshold, double simplifyTolerance, OGRSpatialReference* targetSRS);

   void convertDsmToShpSrs(const std::string& dsmFilePath, const std::string& Shpfile_trans_Name, const std::string& outputDsmFilePath);
    std::vector<std::pair<std::vector<projection::Point_2>, std::pair<double, double>>> extractBuildingFootprints(const std::string& shpFilePath, const std::string& outputDsmFilePath);


    static std::vector<std::pair<std::vector<projection::Point_2>, std::pair<double, double>>> readShpFile_simplify(const std::string& filePath, const std::string& dsmFilePath);

private:
	void* AllocateMemory(const GCDataType lDataType, const long lSize);   // 智能分配内存
	GCDataType GDALType2GCType(const GDALDataType ty); // GDAL Type ==========> GDALOpenCV Type
	GDALDataType GCType2GDALType(const GCDataType ty); //  GDALOpenCV Type ==========> GDAL Type
	GCDataType OPenCVType2GCType(const int ty); // OPenCV Type ==========> GDALOpenCV Type
	int GCType2OPenCVType(const GCDataType ty); // GDALOpenCV Type ==========> OPenCV Type

    static cv::Mat createMaskFromPolygon(const std::vector<projection::Point_2>& footprint, int rasterWidth, int rasterHeight, double* geoTransform);



};

