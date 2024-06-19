/*建筑物投影，集成时间2023年12月*/
#pragma once
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/algorithm.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/polygon_mesh_processing.h>
#include <boost/iterator/zip_iterator.hpp>
#include <float.h>
#include <vector>
#include "psimpl.h"


namespace projection {

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef Kernel::Point_3                                     Point_3;
    typedef Kernel::Point_2										Point_2;
    typedef CGAL::Surface_mesh<Point_3>							Mesh;

    typedef boost::graph_traits<Mesh>::vertex_descriptor							vertex_descriptor;
    typedef boost::graph_traits<Mesh>::face_descriptor							    face_descriptor;
    typedef boost::graph_traits<Mesh>::halfedge_descriptor							halfedge_descriptor;

    typedef CGAL::Search_traits_3<Kernel>                       Traits;
    typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
    typedef K_neighbor_search::Tree                             Tree;

    typedef Kernel::FT                                          FT;
    typedef Kernel::Segment_2                                   Segment;
    typedef CGAL::Alpha_shape_vertex_base_2<Kernel>             Vb;
    typedef CGAL::Alpha_shape_face_base_2<Kernel>               Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb, Fb>        Tds;
    typedef CGAL::Delaunay_triangulation_2<Kernel, Tds>         Triangulation_2;
    typedef CGAL::Alpha_shape_2<Triangulation_2>                Alpha_shape_2;
    typedef Alpha_shape_2::Alpha_shape_edges_iterator           Alpha_shape_edges_iterator;
    namespace PMP = CGAL::Polygon_mesh_processing;

    //输入建筑物实例网格模型，输出Lod1网格模型
    class buildingLod1Mesh
    {
    private:
        Mesh m;//输入原始mesh模型
        double height_z;//mesh模型最高点
        double lowest_z;//mesh模型最低点
        std::vector<Point_2> simplfyPoints;//墙面足迹轮廓点
        std::map<Point_3, Mesh::Vertex_index> pointToVertexMap;  // Point to Vertex mapping

    protected:

        //alpha shape分割
        template <class OutputIterator>
        void alphaEdges(const Alpha_shape_2& A, OutputIterator out);

        void removeNoise(std::vector<Point_3>& oldPointSets, std::vector<Point_3>& newPointSets, double ratio, std::vector<double> pz);

        void getAlphaShape(const std::vector<Point_2>& pointSet, std::vector<Point_2>& contour, int para);

        void simplify(const std::vector<Point_2>& contour, std::vector<Point_2>& simplified_contour);

    public:
        buildingLod1Mesh(const std::vector<Point_2>& points, double lowest , double height)
            : simplfyPoints(points), lowest_z(lowest), height_z(height) {}
        ~buildingLod1Mesh() {}

        void buildMesh(Mesh& mesh);
    };

    class pointcloudProjection {
    private:
        std::map<int, std::vector<Point_3>> pointSets;

    public:
        pointcloudProjection(std::map<int, std::vector<Point_3>> pointsets);
        void projection(std::map<int, std::vector<Point_3>>& newPointSets, int neightborPoint = 50, int groudPointTag = 0);
    };
}


//
//
//        //搜索点云密度，剔除稀疏点云
//        void removeNoise(std::vector<Point_3>& oldPointSets, std::vector<Point_3>& newPointSets, double ratio, std::vector<double>pz)
//        {
//            std::vector<double>radius;//50个临近点的半径
//            Tree tree(oldPointSets.begin(), oldPointSets.end());
//            double sumRadius = 0;
//            double sum_z = 0;
//            for (double pointz : pz)
//            {
//                sum_z += pointz;
//            }
//            for (Point_3 searchPoint : oldPointSets)
//            {
//
//                K_neighbor_search search(tree, searchPoint, 50);
//                double distance = 0;//50个点平均距离
//                for (K_neighbor_search::iterator it = search.begin(); it != search.end(); ++it)
//                {
//                    distance += it->second;
//                }
//                distance /= 50;
//                sumRadius += distance;
//                radius.push_back(distance);
//            }
//
//            int pointNum = oldPointSets.size();
//            double averageRadius = sumRadius / pointNum;
//            double averageZ = sum_z / pointNum;
//            double scalRate = (height_z - lowest_z) / averageRadius;
//            for (int index = 0; index < pointNum; index++)
//            {
//                if ((radius[index] + 0.5 * (pz[index] - lowest_z) / scalRate) < ratio * (averageRadius + 0.5 * (averageZ - lowest_z) / scalRate))
//                    newPointSets.push_back(oldPointSets[index]);
//            }
//        }
//
//        //输入一圈点云，输出alpha shape
//        void getAlphaShape(const std::vector<Point_2>& pointSet, std::vector<Point_2>& contour, int para)
//        {
//            Alpha_shape_2 A(pointSet.begin(), pointSet.end(), FT(para), Alpha_shape_2::GENERAL);
//            std::vector<Segment> segments;
//            alphaEdges(A, std::back_inserter(segments));
//            std::map<Point_2, int> flagp;
//            int num = 0;
//            for (auto& seg : segments) {
//                Point_2 p1(seg.vertex(0).x(), seg.vertex(0).y());
//                Point_2 p2(seg.vertex(1).x(), seg.vertex(1).y());
//                if (flagp[p1] == 0)
//                {
//                    flagp[p1] = ++num;
//                }
//                if (flagp[p2] == 0)
//                {
//                    flagp[p2] = ++num;
//                }
//            }
//            std::vector<std::vector< Point_2>> allcontour;
//            while (segments.size() != 0)
//            {
//                std::vector<Point_2> currentcontour;
//                int headid = flagp[segments.begin()->vertex(1)];
//                int endid = flagp[segments.begin()->vertex(0)];
//                currentcontour.push_back(segments.begin()->vertex(0));
//                segments.erase(segments.begin());
//                for (auto iter = segments.begin(); iter != segments.end(); )
//                {
//                    if (endid == flagp[iter->vertex(0)])
//                    {
//                        break;
//                    }
//                    if (headid == flagp[iter->vertex(0)])
//                    {
//                        currentcontour.push_back(iter->vertex(0));
//                        headid = flagp[iter->vertex(1)];
//                        segments.erase(iter);
//                        iter = segments.begin();
//                        continue;
//                    }
//                    else
//                    {
//                        iter++;
//                    }
//                }
//                allcontour.push_back(currentcontour);
//            }
//            int maxnum = 0;
//            int id_max = 0;
//            for (int j = 0; j < allcontour.size(); j++)
//            {
//                if (allcontour[j].size() > maxnum)
//                {
//                    id_max = j;
//                    maxnum = allcontour[j].size();
//                }
//            }
//
//            for (int j = 0; j < allcontour[id_max].size(); j++)
//            {
//                contour.push_back(allcontour[id_max][j]);
//            }
//        }
//
//        //alpha shape 轮廓点简化为足迹点
//        void simplify(const std::vector<Point_2>& contour, std::vector<Point_2>& simplified_contour)
//        {
//            double num = 0;
//
//            std::vector<double> Origin_border, Result_border;
//            for (auto iter1 = contour.begin(); iter1 != contour.end(); iter1++)
//            {
//                Origin_border.push_back(iter1->x());
//                Origin_border.push_back(iter1->y());
//            }
//            psimpl::simplify_douglas_peucker<2>(Origin_border.begin(), Origin_border.end(), 3, std::back_inserter(Result_border));
//
//            for (int i = 0; i < Result_border.size() / 2; ++i)
//            {
//                Point_2 currentpoint(Result_border[2 * i + 0], Result_border[2 * i + 1]);
//                simplified_contour.push_back(currentpoint);
//            }
//        }
//
//    public:
//
//        buildingLod1Mesh(const std::vector<projection::Point_2>& footprint, double height, double ground);
//        {
//
//            height_z = height;
//            lowest_z = ground;
//
//            // 直接使用传入的CGAL点
//            simplfyPoints = footprint;
//        }
//
//        ~buildingLod1Mesh() {}
//
//        //生成Lod1模型
//        void buildMesh(Mesh& mesh)
//        {
//            //插入顶点并保存句柄
//            std::vector<vertex_descriptor>upVertices;
//            std::vector<vertex_descriptor>downVertices;
//            for (Point_2 p2D : simplfyPoints)
//            {
//                vertex_descriptor upVertex = mesh.add_vertex(Point_3(p2D.x(), p2D.y(), height_z));
//                vertex_descriptor downVertex = mesh.add_vertex(Point_3(p2D.x(), p2D.y(), lowest_z));
//                upVertices.push_back(upVertex);
//                downVertices.push_back(downVertex);
//            }
//            //添加顶部
//            mesh.add_face(upVertices);
//            //底部顶点反向
//            std::vector<vertex_descriptor>downVertices1;
//            for (int i = downVertices.size() - 1; i >= 0; i--)
//            {
//                downVertices1.push_back(downVertices[i]);
//            }
//            //添加底部
//            mesh.add_face(downVertices1);
//            //顶点链闭环
//            vertex_descriptor upTail = upVertices[0];
//            upVertices.push_back(upTail);
//            vertex_descriptor downTail = downVertices[0];
//            downVertices.push_back(downTail);
//
//            //添加侧面
//            for (int index = 0; index < upVertices.size() - 1; index++)
//            {
//                mesh.add_face(upVertices[index + 1], upVertices[index], downVertices[index], downVertices[index + 1]);
//            }
//            PMP::triangulate_faces(mesh);
//
//        }
//
//    };
//
//    //输入点云，输出投影到最低点的点云
//    class pointcloudProjection
//    {
//    private:
//        std::map<int, std::vector<Point_3>>pointSets;
//
//    public:
//        pointcloudProjection(std::map<int, std::vector<Point_3>>pointsets) :pointSets(pointsets) {}
//        void projection(std::map<int, std::vector<Point_3>>& newPointSets, int neightborPoint = 50, int groudPointTag = 0)
//        {
//            if (neightborPoint <= 0)
//            {
//                std::cout << "neightborPoint number error with value " << neightborPoint << std::endl;
//                system("pause");
//            }
//            Tree tree(pointSets[groudPointTag].begin(), pointSets[0].end());
//            std::map<int, std::vector<Point_3>>::iterator point_iterator;
//            for (point_iterator = pointSets.begin(); point_iterator != pointSets.end(); point_iterator++)
//            {
//                if (point_iterator->first == groudPointTag)
//                {
//                    std::vector<Point_3>newpoints;
//                    for (Point_3 searchPoint : point_iterator->second)
//                    {
//                        newpoints.push_back(searchPoint);
//                    }
//                    newPointSets[groudPointTag] = newpoints;
//                    continue;
//                }
//
//                std::vector<Point_3>newpoints;
//                for (Point_3 searchPoint : point_iterator->second)
//                {
//                    K_neighbor_search search(tree, searchPoint, neightborPoint);
//                    double height = 0;
//                    for (K_neighbor_search::iterator it = search.begin(); it != search.end(); ++it)
//                    {
//                        height += it->first.hz();
//                    }
//                    height /= neightborPoint;
//                    Point_3 newpoint(searchPoint.hx(), searchPoint.hy(), height);
//                    newpoints.push_back(newpoint);
//                }
//                int tag = point_iterator->first;
//                newPointSets[tag] = newpoints;
//            }
//        }
//    };
//}