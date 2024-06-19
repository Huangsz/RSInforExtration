
#include <ogrsf_frmts.h>
#include <gdal_priv.h>
#include <gdal_alg.h>
#include <iostream>
#include <limits>
#include <gdalwarper.h> 
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

#include "GdalOpencvTensor.h"

GdalOpencvTensor::GdalOpencvTensor()
{
}

GdalOpencvTensor::~GdalOpencvTensor()
{
}

void* GdalOpencvTensor::AllocateMemory(const GCDataType lDataType, const long lSize)
{
    assert(0 != lSize);
    void* pvData = NULL;
    switch (lDataType)
    {
    case GC_Byte:
        pvData = new(std::nothrow) unsigned char[lSize];
        break;
    case GC_UInt16:
        pvData = new(std::nothrow) unsigned short int[lSize];
        break;
    case GC_Int16:
        pvData = new(std::nothrow) short int[lSize];
        break;
    case GC_UInt32:
        pvData = new(std::nothrow) unsigned long[lSize];
        break;
    case GC_Int32:
        pvData = new(std::nothrow) long[lSize];
        break;
    case GC_Float32:
        pvData = new(std::nothrow) float[lSize];
        break;
    case GC_Float64:
        pvData = new(std::nothrow) double[lSize];
        break;
    default:
        assert(false);
        break;
    }
    return pvData;
}

GCDataType GdalOpencvTensor::GDALType2GCType(const GDALDataType ty)
{
    switch (ty)
    {
    case GDT_Byte:
        return GC_Byte;
    case GDT_UInt16:
        return GC_UInt16;
    case GDT_Int16:
        return GC_Int16;
    case GDT_UInt32:
        return GC_UInt32;
    case GDT_Int32:
        return GC_Int32;
    case GDT_Float32:
        return GC_Float32;
    case GDT_Float64:
        return GC_Float64;
    default:
        assert(false);
        return GC_ERRType;
    }
}

GCDataType GdalOpencvTensor::OPenCVType2GCType(const int ty)
{
    switch (ty)
    {
    case 0:
        return GC_Byte;
    case 2:
        return GC_UInt16;
    case 3:
        return GC_Int16;
    case 4:
        return GC_Int32;
    case 5:
        return GC_Float32;
    case 6:
        return GC_Float64;
    default:
        assert(false);
        return GC_ERRType;
    }
}

GDALDataType GdalOpencvTensor::GCType2GDALType(const GCDataType ty)
{
    switch (ty)
    {
    case GC_Byte:
        return GDT_Byte;
    case GC_UInt16:
        return GDT_UInt16;
    case GC_Int16:
        return GDT_Int16;
    case GC_UInt32:
        return GDT_UInt32;
    case GC_Int32:
        return GDT_Int32;
    case GC_Float32:
        return GDT_Float32;
    case GC_Float64:
        return GDT_Float64;
    default:
        assert(false);
        return GDT_TypeCount;
    }
}

int GdalOpencvTensor::GCType2OPenCVType(const GCDataType ty)
{
    switch (ty)
    {
    case GC_Byte:
        return 0;
    case GC_UInt16:
        return 2;
    case GC_Int16:
        return 3;
    case GC_Int32:
        return 4;
    case GC_Float32:
        return 5;
    case GC_Float64:
        return 6;
    default:
        assert(false);
        return -1;
    }
}

tensorflow::Tensor GdalOpencvTensor::Mat2Tensor(cv::Mat& img)   //��opencv������ת��Ϊtensor��������
{
    tensorflow::Tensor image_input = tensorflow::Tensor(tensorflow::DT_FLOAT,
        tensorflow::TensorShape({ 1, img.rows, img.cols, img.channels() }));
    float* tensor_data_ptr = image_input.flat<float>().data();
    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()), tensor_data_ptr);
    img.convertTo(fake_mat, CV_32FC(img.channels()));    //��Mat���͵ľ���ת��ΪCV_32FC���͵ľ���
    return image_input;
    return tensorflow::Tensor();
}

void GdalOpencvTensor::histogramEqualization(GDALRasterBand* poBand, cv::Mat& outputMat) {
    int nXSize = poBand->GetXSize();
    int nYSize = poBand->GetYSize();

    // ��ȡ��������
    std::vector<float> data(nXSize * nYSize);
    poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, data.data(), nXSize, nYSize, GDT_Float32, 0, 0);

    // ת��ΪOpenCV Mat
    cv::Mat mat(nYSize, nXSize, CV_32FC1, data.data());

    // ��һ����0-255��Χ
    cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
    mat.convertTo(mat, CV_8UC1);

    // Ӧ��ֱ��ͼ���⻯
    cv::equalizeHist(mat, outputMat);
}

bool GdalOpencvTensor::GDAL2Mat(std::string input_image_tif, cv::Mat& img)
{
    GDALDataset* m_poDataSet = (GDALDataset*)GDALOpen(input_image_tif.c_str(), GA_ReadOnly);    //GDAL���ݼ�
    if (!m_poDataSet) {
        std::cerr << "�޷����ļ�: " << input_image_tif << std::endl;
        return false;
    }

    // ��ȡӰ��ĸ߶ȡ���ȺͲ�����
    int m_imgHeight = m_poDataSet->GetRasterYSize();
    int m_imgWidth = m_poDataSet->GetRasterXSize();
    int m_bandNum = m_poDataSet->GetRasterCount();

    if (m_bandNum <= 0) {
        std::cerr << "���ݼ���û�в��Ρ�" << std::endl;
        GDALClose(m_poDataSet);
        return false;
    }

    // ��ȡ��һ�����β�ȷ����������
        // ��ȡ��һ�����β�ȷ����������
    GDALRasterBand* pBand = m_poDataSet->GetRasterBand(1);
    GDALDataType gdalType = pBand->GetRasterDataType();
    GCDataType m_dataType = GDALType2GCType(gdalType);         //GDAL Type ==========> GDALOpenCV Type
    int cvType = GCType2OPenCVType(m_dataType);
    if (cvType == -1) {
        GDALClose(m_poDataSet);
        return false;
    }

    // �������ڴ洢�������ݵ�����
    std::vector<cv::Mat> imgBands(m_bandNum);
    // �����ڴ滺���������ڶ�ȡ��������
    std::unique_ptr<void, void(*)(void*)> buffer(CPLMalloc(GDALGetDataTypeSizeBytes(gdalType) * m_imgWidth * m_imgHeight), CPLFree);


    // ����ÿ������
    for (int iBand = 0; iBand < m_bandNum; ++iBand) {
        pBand = m_poDataSet->GetRasterBand(iBand + 1);
        // ��ȡ�������ݵ�������
        if (pBand->RasterIO(GF_Read, 0, 0, m_imgWidth, m_imgHeight, buffer.get(), m_imgWidth, m_imgHeight, gdalType, 0, 0) != CE_None) {
            std::cerr << "��ȡ���� " << iBand + 1 << " ʧ�ܡ�" << std::endl;
            GDALClose(m_poDataSet);
            return false;
        }
        // ������������ת��ΪOpenCV���󣬲���¡��������
        imgBands[iBand] = cv::Mat(m_imgHeight, m_imgWidth, cvType, buffer.get()).clone();
    }

    // �ϲ����в����γ����յĲ�ɫͼ��
    cv::merge(imgBands, img);
    // �ر�GDAL���ݼ�
    GDALClose(m_poDataSet);
    return true;
}


bool GdalOpencvTensor::TIF2JPG(std::string input_image_tif, std::string input_image_tif_float, std::string outputFilename)
{
    GDALDataset* m_poDataSet = (GDALDataset*)GDALOpen(input_image_tif.c_str(), GA_ReadOnly);    //GDAL���ݼ�
    if (!m_poDataSet) {
        std::cerr << "�޷����ļ�: " << input_image_tif << std::endl;
        return false;
    }

    int m_imgHeight = m_poDataSet->GetRasterYSize(); // Ӱ������
    int m_imgWidth = m_poDataSet->GetRasterXSize();  // Ӱ������
    int m_bandNum = m_poDataSet->GetRasterCount();   // Ӱ�񲨶���


    GDALRasterBand* pBand = m_poDataSet->GetRasterBand(1);
    GDALDataType gdalType = pBand->GetRasterDataType(); // ��ȡӰ����������

    // ��ȡ����任������ͶӰ��Ϣ
    double geoTransform[6];
    m_poDataSet->GetGeoTransform(geoTransform);
    const char* projectionRef = m_poDataSet->GetProjectionRef();

    // �����µ�GDAL���ݼ����ڴ洢����������
    GDALDriver* poDriver_new = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* poDstDataset = poDriver_new->Create(input_image_tif_float.c_str(), m_imgWidth, m_imgHeight, m_bandNum, GDT_Float32, nullptr);
    if (!poDstDataset) {
        std::cerr << "�޷��������TIFF�ļ�: " << input_image_tif_float << std::endl;
        GDALClose(m_poDataSet);
        return false;
    }

    // ���õ���任������ͶӰ��Ϣ
    poDstDataset->SetGeoTransform(geoTransform);
    poDstDataset->SetProjection(projectionRef);

    // ����ÿ������
    for (int iBand = 1; iBand <= m_bandNum; ++iBand) {
        GDALRasterBand* poSrcBand = m_poDataSet->GetRasterBand(iBand);
        GDALRasterBand* poDstBand = poDstDataset->GetRasterBand(iBand);

        // �����ڴ����ڶ�ȡ�ʹ洢����
        std::vector<uint16_t> pData(m_imgWidth * m_imgHeight);
        std::vector<float> pFloatData(m_imgWidth * m_imgHeight);

        // ��ȡ����
        poSrcBand->RasterIO(GF_Read, 0, 0, m_imgWidth, m_imgHeight, pData.data(), m_imgWidth, m_imgHeight, GDT_UInt16, 0, 0);

        // ת����������
        std::transform(pData.begin(), pData.end(), pFloatData.begin(), [](uint16_t val) { return static_cast<float>(val); });

        // д������
        poDstBand->RasterIO(GF_Write, 0, 0, m_imgWidth, m_imgHeight, pFloatData.data(), m_imgWidth, m_imgHeight, GDT_Float32, 0, 0);
    }

    // ����OpenCV����洢����������
    std::vector<cv::Mat> bands;
    for (int iBand = 1; iBand <= m_bandNum; ++iBand) {
        GDALRasterBand* poBand_new = poDstDataset->GetRasterBand(iBand);
        cv::Mat band(m_imgHeight, m_imgWidth, CV_32FC1);
        histogramEqualization(poBand_new, band); // ֱ��ͼ���⻯
        bands.push_back(band);
    }

    // �ϲ����β�ת��Ϊ8λͼ��
    cv::Mat merged;
    cv::Mat img_merge[3] = { bands[2], bands[1], bands[0] };// BGR
    cv::merge(img_merge, 3, merged);
    merged.convertTo(merged, CV_8UC3);

    // ����ΪJPEG�ļ�
    cv::imwrite(outputFilename, merged);

    // �ͷ�GDAL���ݼ�
    GDALClose(m_poDataSet);
    GDALClose(poDstDataset);

    return true;
}

void GdalOpencvTensor::tif2shp(GDALDataset* poDataset, GDALDataset* poDataset_prj_ref, std::string outFilePath)
{
    GDALAllRegister(); // ע��GDAL����
    OGRRegisterAll();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO"); // �������·����������

    GDALRasterBand* band1 = poDataset->GetRasterBand(1);
    GDALDataType datatype = band1->GetRasterDataType();

    // ��ȡShapefile����
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    if (!poDriver) {
        std::cerr << "ESRI Shapefile���������á�" << std::endl;
        return;
    }

    // �������ʸ�����ݼ�
    GDALDataset* poDstDS = poDriver->Create(outFilePath.c_str(), 0, 0, 0, datatype, nullptr);
    if (!poDstDS) {
        std::cerr << "�޷��������Shapefile: " << outFilePath << std::endl;
        return;
    }

    // ����ռ�ο���������Ӱ����ͬ
    const char* projectionRef = poDataset_prj_ref->GetProjectionRef();
    OGRSpatialReference oSRS(projectionRef);
    OGRLayer* poLayer = poDstDS->CreateLayer("DstLayer", &oSRS, wkbPolygon, nullptr);
    if (!poLayer) {
        std::cerr << "����ͼ��ʧ�ܡ�" << std::endl;
        GDALClose(poDstDS);
        return;
    }

    // �������Ա��ֶ�
    OGRFieldDefn oField("gridcode", OFTInteger);
    if (poLayer->CreateField(&oField) != OGRERR_NONE) {
        std::cerr << "���������ֶ�ʧ�ܡ�" << std::endl;
        GDALClose(poDstDS);
        return;
    }

    // ����դ��ʸ����
    GDALRasterBandH hSrcBand = (GDALRasterBandH)band1;
    if (GDALPolygonize(hSrcBand, nullptr, (OGRLayerH)poLayer, 0, nullptr, nullptr, nullptr) != CE_None) {
        std::cerr << "դ��ʸ����ʧ�ܡ�" << std::endl;
        GDALClose(poDstDS);
        return;
    }

    // ɾ�� gridcode Ϊ 0 ��Ҫ��
    poLayer->SetAttributeFilter("gridcode = 0");
    OGRFeature* po_feat = nullptr;
    while ((po_feat = poLayer->GetNextFeature()) != nullptr) {
        poLayer->DeleteFeature(po_feat->GetFID());
        OGRFeature::DestroyFeature(po_feat);
    }

    // ִ�� REPACK ����
    std::string sql = "REPACK " + std::string(poLayer->GetName());
    poDstDS->ExecuteSQL(sql.c_str(), nullptr, nullptr);

    // ���� PRJ �ļ�
    std::string prjFilename = outFilePath.substr(0, outFilePath.find_last_of('.')) + ".prj";
    char* pszWKT = nullptr;
    oSRS.exportToWkt(&pszWKT);
    std::ofstream prjFile(prjFilename);
    if (prjFile.is_open()) {
        prjFile << pszWKT;
        prjFile.close();
    }
    else {
        std::cerr << "�޷�����PRJ�ļ�: " << prjFilename << std::endl;
    }
    CPLFree(pszWKT);

    // �ر��ļ�
    GDALClose(poDstDS);
}


GDALDataset* GdalOpencvTensor::save_result_tif(GDALDataset* m_poDataSet, const std::string& Result_tiffile_Name, double* adfGeoTransform, tensorflow::uint8* result_tensor)
{
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("Gtiff");
    if (poDriver == nullptr) {
        std::cerr << "GTiff���������á�" << std::endl;
        return nullptr;
    }

    // �������TIFF�ļ�
    GDALDataset* poDataset2 = poDriver->Create(Result_tiffile_Name.c_str(), m_poDataSet->GetRasterXSize(), m_poDataSet->GetRasterYSize(), 1, GDT_Byte, nullptr);
    if (!poDataset2) {
        std::cerr << "�޷��������TIFF�ļ�: " << Result_tiffile_Name << std::endl;
        return nullptr;
    }

    // ���õ���任����
    poDataset2->SetGeoTransform(adfGeoTransform);

    // ���������������
    cv::Mat img_merge_out(cv::Size(m_poDataSet->GetRasterXSize(), m_poDataSet->GetRasterYSize()), CV_8UC1, result_tensor);

    // д�������ݵ�TIFF�ļ�
    GDALRasterBand* pBand = poDataset2->GetRasterBand(1);
    pBand->RasterIO(GF_Write, 0, 0, m_poDataSet->GetRasterXSize(), m_poDataSet->GetRasterYSize(), img_merge_out.data, m_poDataSet->GetRasterXSize(), m_poDataSet->GetRasterYSize(), GDT_Byte, 0, 0);

    return poDataset2;
}


void GdalOpencvTensor::readShapefile(const char* filename, std::vector<OGRPolygon*>& polygons, OGRSpatialReference** sourceSRS) {
    GDALAllRegister();
    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(filename, GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        std::cerr << "Failed to open shapefile: " << filename << std::endl;
        return;
    }
    OGRLayer* poLayer = poDS->GetLayer(0);
    if (poLayer == nullptr) {
        std::cerr << "Failed to get layer from shapefile: " << filename << std::endl;
        GDALClose(poDS);
        return;
    }

    *sourceSRS = poLayer->GetSpatialRef();
    std::string prjFilePath = std::string(filename);
    prjFilePath.replace(prjFilePath.end() - 3, prjFilePath.end(), "prj");
    if (!importSRSFromPrj(prjFilePath.c_str(), *sourceSRS)) {
        std::cerr << "No spatial reference found in shapefile and failed to import from PRJ file." << std::endl;
        GDALClose(poDS);
        return;
    }
    char* wkt = nullptr;
    (*sourceSRS)->exportToWkt(&wkt);
    std::cout << "Source Spatial Reference (WKT): " << wkt << std::endl;
    OGRFeature* poFeature;
    poLayer->ResetReading();
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        OGRGeometry* poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            OGRPolygon* poPolygon = (OGRPolygon*)poGeometry->clone();
            polygons.push_back( poPolygon);
        }
        OGRFeature::DestroyFeature(poFeature);
    }
    GDALClose(poDS);
}

double  GdalOpencvTensor::calculatePolygonArea(std::vector<OGRPoint>& polygon_Data, double* adfGeoTransform) {
    double area = 0.0;
    for (size_t i = 0; i < polygon_Data.size(); ++i) {
        OGRPoint& p1 = polygon_Data[i];
        OGRPoint& p2 = polygon_Data[(i + 1) % polygon_Data.size()];
        // ����������ת��Ϊ��������
        double x1 = adfGeoTransform[0] + p1.getX() * adfGeoTransform[1] + p1.getY() * adfGeoTransform[2];
        double y1 = adfGeoTransform[3] + p1.getX() * adfGeoTransform[4] + p1.getY() * adfGeoTransform[5];
        double x2 = adfGeoTransform[0] + p2.getX() * adfGeoTransform[1] + p2.getY() * adfGeoTransform[2];
        double y2 = adfGeoTransform[3] + p2.getX() * adfGeoTransform[4] + p2.getY() * adfGeoTransform[5];
        area += (x1 * y2 - x2 * y1);
    }
    return std::abs(area) / 2.0;
}

int GdalOpencvTensor::getUTMZone(double longitude) {
    return static_cast<int>(std::floor((longitude + 180) / 6)) + 1;
}

void GdalOpencvTensor::transformToUTM(std::vector<OGRPolygon*>& polygons, OGRSpatialReference* sourceSRS, OGRSpatialReference*& targetSRS) {
    if (polygons.empty()) {
        std::cerr << "No polygons found in the shapefile." << std::endl;
        return;
    }
    if (sourceSRS == nullptr) {
        std::cerr << "No source spatial reference system found." << std::endl;
        return;
    }
    // ʹ�õ�һ������ε����ĵ������� UTM ����
    OGREnvelope envelope;
    polygons[0]->getEnvelope(&envelope);
    double longitude = (envelope.MinX + envelope.MaxX) / 2.0;
    double latitude = (envelope.MinY + envelope.MaxY) / 2.0;

    // ��֤γ��ֵ
    if (latitude < -80.0 || latitude > 84.0) {
        std::cerr << "Invalid latitude for UTM zone calculation: " << latitude << std::endl;
        return;
    }

    int utmZone = getUTMZone(longitude);
    bool isNorthernHemisphere = (latitude >= 0.0);

    std::cout << "UTM Zone: " << utmZone << (isNorthernHemisphere ? "N" : "S") << std::endl;

    targetSRS = new OGRSpatialReference();
    if (targetSRS->SetWellKnownGeogCS("WGS84") != OGRERR_NONE) {
        std::cerr << "Failed to set geographic coordinate system to WGS84." << std::endl;
        return;
    }
    if (targetSRS->SetUTM(utmZone, isNorthernHemisphere) != OGRERR_NONE) {
        std::cerr << "Failed to set UTM zone: " << utmZone << " for " << (isNorthernHemisphere ? "Northern" : "Southern") << " Hemisphere." << std::endl;
        return;
    }
    // ����ͶӰ����ϵ����
    std::string projCSName = "UTM Zone " + std::to_string(utmZone) + (isNorthernHemisphere ? "N" : "S");
    targetSRS->SetProjCS(projCSName.c_str());
    // ���Ŀ��ռ�ο�ϵ�� WKT
    char* targetWKT = nullptr;
    targetSRS->exportToWkt(&targetWKT);
    std::cout << "Target Spatial Reference (WKT): " << targetWKT << std::endl;
    CPLFree(targetWKT);

    OGRCoordinateTransformation* transform = OGRCreateCoordinateTransformation(sourceSRS, targetSRS);
    if (transform == NULL) {
        std::cerr << "Failed to create coordinate transformation." << std::endl;
        return;
    }
    OCTDestroyCoordinateTransformation(transform);
}


bool GdalOpencvTensor::importSRSFromPrj(const char* prjFilePath, OGRSpatialReference*& sourceSRS) {
    std::ifstream prjFile(prjFilePath);
    if (!prjFile.is_open()) {
        std::cerr << "Failed to open PRJ file: " << prjFilePath << std::endl;
        return false;
    }
    std::stringstream buffer;
    buffer << prjFile.rdbuf();
    std::string wkt = buffer.str();
    prjFile.close();
    sourceSRS = new OGRSpatialReference();
    if (sourceSRS->importFromWkt(wkt.c_str()) != OGRERR_NONE) {
        std::cerr << "Failed to import spatial reference from WKT." << std::endl;
        delete sourceSRS;
        sourceSRS = nullptr;
        return false;
    }
    std::cout << "Successfully imported spatial reference from PRJ file." << std::endl;
    return true;
}

void GdalOpencvTensor::writePolygonShapefile(const char* outputShapefile, std::vector<OGRPolygon*>& polygons, double areaThreshold, double simplifyTolerance, OGRSpatialReference* targetSRS) {
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    if (poDriver == NULL) {
        std::cerr << "Shapefile driver not available." << std::endl;
        return;
    }
    GDALDataset* poDS = poDriver->Create(outputShapefile, 0, 0, 0, GDT_Unknown, NULL);
    if (poDS == NULL) {
        std::cerr << "Failed to create shapefile: " << outputShapefile << std::endl;
        return;
    }
    OGRLayer* poLayer = poDS->CreateLayer("polygons", targetSRS, wkbPolygon, NULL);
    if (poLayer == NULL) {
        std::cerr << "Failed to create layer." << std::endl;
        GDALClose(poDS);
        return;
    }
    for (const auto& polygonData : polygons) {
        // ת������ε�Ŀ������ϵ
        OGRGeometry* transformedGeometry = polygonData->clone();
        transformedGeometry->transformTo(targetSRS);
        // ����ת����Ķ�������
        OGRPolygon* transformedPolygon = dynamic_cast<OGRPolygon*>(transformedGeometry);
        if (transformedPolygon) {
            double area = transformedPolygon->get_Area();
            if (area >= areaThreshold) {
                // �Զ���ν��м�
                OGRGeometry* simplifiedGeometry = transformedPolygon->SimplifyPreserveTopology(simplifyTolerance);
                if (simplifiedGeometry != nullptr && simplifiedGeometry->IsValid()) {
                    OGRFeature* poFeature = OGRFeature::CreateFeature(poLayer->GetLayerDefn());
                    poFeature->SetGeometry(simplifiedGeometry);
                    if (poLayer->CreateFeature(poFeature) != OGRERR_NONE) {
                        std::cerr << "Failed to create feature in shapefile." << std::endl;
                    }
                    OGRFeature::DestroyFeature(poFeature);
                    OGRGeometryFactory::destroyGeometry(simplifiedGeometry);
                }
                else {
                    std::cerr << "Failed to simplify polygon or simplified polygon is invalid." << std::endl;
                }
            }
        }
        else {
            std::cerr << "Transformed geometry is not a polygon." << std::endl;
        }
        OGRGeometryFactory::destroyGeometry(transformedGeometry);
    }
    GDALClose(poDS);
    //д�� .prj �ļ�
    std::string prjFilename(outputShapefile);
    prjFilename.replace(prjFilename.end() - 3, prjFilename.end(), "prj");
    char* wkt = nullptr;
    if (targetSRS->exportToWkt(&wkt) == OGRERR_NONE) {
        std::ofstream prjFile(prjFilename);
        if (prjFile.is_open()) {
            prjFile << wkt;
            prjFile.close();
        }
        else {
            std::cerr << "Failed to create .prj file: " << prjFilename << std::endl;
        }
        CPLFree(wkt);
    }
    else {
        std::cerr << "Failed to export spatial reference to WKT." << std::endl;
    }
}

void GdalOpencvTensor::convertDsmToShpSrs(const std::string& dsmFilePath, const std::string& Shpfile_trans_Name, const std::string& outputDsmFilePath) {
    OGRSpatialReference shpSRS;
    std::string prjFilePath = Shpfile_trans_Name.substr(0, Shpfile_trans_Name.find_last_of('.')) + ".prj";
    char** prjFileText = CSLLoad(prjFilePath.c_str());
    if (shpSRS.importFromESRI(prjFileText) != OGRERR_NONE) {
        std::cerr << "Failed to import spatial reference from PRJ file." << std::endl;
        CSLDestroy(prjFileText);
        return;
    }
    CSLDestroy(prjFileText);

    GDALDataset* dsmDS = (GDALDataset*)GDALOpen(dsmFilePath.c_str(), GA_ReadOnly);
    if (dsmDS == NULL) {
        std::cerr << "Failed to open DSM file: " << dsmFilePath << std::endl;
        return;
    }

    const char* dsmWkt = dsmDS->GetProjectionRef();
    OGRSpatialReference dsmSRS;
    if (dsmSRS.importFromWkt(dsmWkt) != OGRERR_NONE) {
        std::cerr << "Failed to import DSM spatial reference from WKT." << std::endl;
        GDALClose(dsmDS);
        return;
    }

    // ���� GDALWarpOptions
    GDALWarpOptions* psWarpOptions = GDALCreateWarpOptions();
    if (psWarpOptions == NULL) {
        std::cerr << "Failed to create GDALWarpOptions." << std::endl;
        GDALClose(dsmDS);
        return;
    }

    psWarpOptions->hSrcDS = dsmDS;
    psWarpOptions->hDstDS = NULL;
    psWarpOptions->nBandCount = 1;
    psWarpOptions->panSrcBands = (int*)CPLMalloc(sizeof(int) * psWarpOptions->nBandCount);
    psWarpOptions->panSrcBands[0] = 1;
    psWarpOptions->panDstBands = (int*)CPLMalloc(sizeof(int) * psWarpOptions->nBandCount);
    psWarpOptions->panDstBands[0] = 1;
    psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

    // ����ͶӰת��
    char* dstWkt = nullptr;
    shpSRS.exportToWkt(&dstWkt);
    psWarpOptions->pTransformerArg = GDALCreateGenImgProjTransformer(dsmDS, dsmWkt, NULL, dstWkt, FALSE, 0.0, 0);
    //CPLFree(dstWkt);

    if (psWarpOptions->pTransformerArg == NULL) {
        std::cerr << "Failed to create transformer argument." << std::endl;
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    // ����Ŀ��ķ���任����
    int rasterWidth = dsmDS->GetRasterXSize();
    int rasterHeight = dsmDS->GetRasterYSize();
    double adfDstGeoTransform[6];
    if (GDALSuggestedWarpOutput(dsmDS, psWarpOptions->pfnTransformer, psWarpOptions->pTransformerArg, adfDstGeoTransform, &rasterWidth, &rasterHeight) == CE_Failure) {
        std::cerr << "Failed to get suggested warp output." << std::endl;
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    // ʹ���ڴ����ݼ�����ת����� DSM ���ݼ�
    GDALDriver* poMemDriver = GetGDALDriverManager()->GetDriverByName("MEM");
    if (poMemDriver == NULL) {
        std::cerr << "MEM driver not available." << std::endl;
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    GDALDataset* memDS = poMemDriver->Create("", rasterWidth, rasterHeight, 1, GDT_Float32, NULL);
    if (memDS == NULL) {
        std::cerr << "Failed to create memory dataset." << std::endl;
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    memDS->SetGeoTransform(adfDstGeoTransform);
    memDS->SetProjection(dstWkt);

    // ִ��ʵ�ʵ�ͶӰת��
    if (GDALReprojectImage(dsmDS, dsmWkt, memDS, dstWkt, GRA_NearestNeighbour, 0.0, 0.0, NULL, NULL, psWarpOptions) != CE_None) {
        std::cerr << "Failed to reproject DSM dataset." << std::endl;
        GDALClose(memDS);
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    // ���ڴ����ݼ����浽�ļ���
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (poDriver == NULL) {
        std::cerr << "GTiff driver not available." << std::endl;
        GDALClose(memDS);
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    GDALDataset* outDS = poDriver->CreateCopy(outputDsmFilePath.c_str(), memDS, FALSE, NULL, NULL, NULL);
    if (outDS == NULL) {
        std::cerr << "Failed to create output DSM file." << std::endl;
        GDALClose(memDS);
        GDALDestroyWarpOptions(psWarpOptions);
        GDALClose(dsmDS);
        return;
    }

    // ���ת���� DSM ���ݼ������귶Χ
    double minX = adfDstGeoTransform[0];
    double maxY = adfDstGeoTransform[3];
    double maxX = minX + rasterWidth * adfDstGeoTransform[1];
    double minY = maxY + rasterHeight * adfDstGeoTransform[5];

    std::cout << "Transformed DSM Dataset Coordinate Range:" << std::endl;
    std::cout << "Min X: " << minX << ", Max X: " << maxX << std::endl;
    std::cout << "Min Y: " << minY << ", Max Y: " << maxY << std::endl;

    GDALClose(outDS);
    GDALClose(memDS);
    GDALDestroyWarpOptions(psWarpOptions);
    GDALClose(dsmDS);
}

std::vector<std::pair<std::vector<projection::Point_2>, std::pair<double, double>>> GdalOpencvTensor::extractBuildingFootprints(const std::string& shpFilePath, const std::string& outputDsmFilePath) {
    std::vector<std::pair<std::vector<projection::Point_2>, std::pair<double, double>>> buildingFootprints;

    GDALAllRegister();
    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(shpFilePath.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        std::cerr << "Open failed." << std::endl;
        exit(1);
    }
    GDALDataset* transformedDsmDS = (GDALDataset*)GDALOpen(outputDsmFilePath.c_str(), GA_ReadOnly);

    // ��TFW�ļ���ȡArcPy��GeoTransform����
    std::string TfwFilePath = outputDsmFilePath.substr(0, outputDsmFilePath.find_last_of('.')) + ".tfw";
    std::ifstream tfwFile(TfwFilePath); // �滻Ϊ���TFW�ļ�·��
    std::vector<double> arcpyGeoTransform(6);
    for (int i = 0; i < 6; ++i) {
        tfwFile >> arcpyGeoTransform[i];
    }

    // ��ȡת����� DSM ���ݼ��Ļ�����Ϣ
    double geoTransform_DSM[6];
    //geoTransform_DSM[0] = arcpyGeoTransform[4]; // top left x
    //geoTransform_DSM[1] = arcpyGeoTransform[0]; // pixel size in x direction
    //geoTransform_DSM[2] = arcpyGeoTransform[1]; // rotation, 0 if image is "north up"
    //geoTransform_DSM[3] = arcpyGeoTransform[5]; // top left y
    //geoTransform_DSM[4] = arcpyGeoTransform[2]; // rotation, 0 if image is "north up"
    //geoTransform_DSM[5] = arcpyGeoTransform[3]; // pixel size in y direction, negative
    if (transformedDsmDS->GetGeoTransform(geoTransform_DSM) != CE_None) {
        std::cerr << "Failed to get geotransform from transformed DSM dataset." << std::endl;
        GDALClose(transformedDsmDS);
        GDALClose(poDS);
        exit(1);
    }

    int rasterWidth = transformedDsmDS->GetRasterXSize();
    int rasterHeight = transformedDsmDS->GetRasterYSize();

    // ��� DSM ���ݼ��ĳߴ��Խ��е���
    std::cout << "DSM Width: " << rasterWidth << ", DSM Height: " << rasterHeight << std::endl;

    OGRLayer* poLayer = poDS->GetLayer(0);
    if (poLayer == NULL) {
        std::cerr << "Failed to get layer from Shapefile." << std::endl;
        GDALClose(transformedDsmDS);
        GDALClose(poDS);
        exit(1);
    }
    double minHeight;
    poLayer->ResetReading();
    OGRFeature* poFeature;
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        OGRGeometry* poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            OGRPolygon* poPolygon = (OGRPolygon*)poGeometry->clone();
            OGRLinearRing* poExteriorRing = poPolygon->getExteriorRing();
            std::vector<projection::Point_2> footprint;

            for (int i = 0; i < poExteriorRing->getNumPoints(); i++) {
                projection::Point_2 point(poExteriorRing->getX(i), poExteriorRing->getY(i));
                footprint.push_back(point);
            }

            // ������Ĥ
            cv::Mat mask = GdalOpencvTensor::createMaskFromPolygon(footprint, rasterWidth, rasterHeight, geoTransform_DSM);

            // ��ȡDSM����
            std::vector<float> dsmData(rasterWidth * rasterHeight);
            float* pData = new float[rasterWidth * rasterHeight];

            GDALRasterBand* poBand = transformedDsmDS->GetRasterBand(1);
            if (poBand == NULL) {
                std::cerr << "Failed to get raster band from transformed DSM dataset." << std::endl;
                GDALClose(transformedDsmDS);
                GDALClose(poDS);
                exit(1);
            }
            GDALDataType dataType = poBand->GetRasterDataType();
            if (GDALRasterIO(poBand, GF_Read, 0, 0, rasterWidth, rasterHeight, pData, rasterWidth, rasterHeight, GDT_Float32, sizeof(float), rasterWidth * sizeof(float)) != CE_None) {
                std::cerr << "Failed to read raster data from transformed DSM dataset." << std::endl;
                GDALClose(transformedDsmDS);
                GDALClose(poDS);
                exit(1);
            }
            // ���������ڵ���Сֵ�����ֵ
            minHeight = std::numeric_limits<double>::max();
            double maxHeight = std::numeric_limits<double>::lowest();
            //double minHeight;
            for (int y = 0; y < rasterHeight; ++y) {
                for (int x = 0; x < rasterWidth; ++x) {
                    float value;
                    // �����������ʹ�pBuffer�л�ȡ����ֵ
                    double dsmValue = pData[y * rasterWidth + x];
                    if (dsmValue < minHeight) {
                        minHeight = dsmValue;
                    }
                    if (mask.at<uchar>(y, x) > 0) {
                        if (dsmValue > maxHeight) {
                            maxHeight = dsmValue;
                        }
                    }
                }
            }

            // ���δ�ҵ���Ч�ĸ߶�ֵ�����������Ϣ
            if (minHeight == std::numeric_limits<double>::max() || maxHeight == std::numeric_limits<double>::lowest()) {
                std::cerr << "No valid height values found for polygon ID: " << poFeature->GetFID() << std::endl;
            }
            else {
                // ������ֵ�����ֵ
                //std::cout << "Polygon ID: " << poFeature->GetFID() << " Min Height: " << minHeight << " Max Height: " << maxHeight << std::endl;
                buildingFootprints.push_back(std::make_pair(footprint, std::make_pair(minHeight, maxHeight)));
            }
        }
        OGRFeature::DestroyFeature(poFeature);
    }

    // Apply the offsets
    //��������ת����������ԭ���ƶ������½ǣ����߶�ֵ�ƶ���������Χ�ڣ��Ա��������
    double x0 = -geoTransform_DSM[0];
    double y0 = -geoTransform_DSM[3];
    double z0 = -minHeight;

    for (auto& footprint : buildingFootprints) {
        for (auto& point : footprint.first) {
            point = projection::Point_2(point.x() + x0, point.y() + y0);
        }
        footprint.second.first += z0;
        footprint.second.second += z0;
    }

    GDALClose(poDS);
    GDALClose(transformedDsmDS);
    return buildingFootprints;
}


cv::Mat GdalOpencvTensor::createMaskFromPolygon(const std::vector<projection::Point_2>& footprint, int rasterWidth, int rasterHeight, double* geoTransform) {
    cv::Mat mask = cv::Mat::zeros(rasterHeight, rasterWidth, CV_8UC1);

    std::vector<cv::Point> cvPoints;
    for (const auto& point : footprint) {
        int x = static_cast<int>((point.x() - geoTransform[0]) / geoTransform[1]);
        int y = static_cast<int>(-(geoTransform[3] - point.y()) / geoTransform[5]);
        cvPoints.push_back(cv::Point(x, y));
    }

    const cv::Point* pts = cvPoints.data();
    int npts = cvPoints.size();
    cv::fillPoly(mask, &pts, &npts, 1, cv::Scalar(255));

    return mask;
}

