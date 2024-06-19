#include "projection.h"
#include "psimpl.h"



namespace projection {

    template <class OutputIterator>
    void buildingLod1Mesh::alphaEdges(const Alpha_shape_2& A, OutputIterator out) {
        Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(),
            end = A.alpha_shape_edges_end();
        for (; it != end; ++it)
            *out++ = A.segment(*it);
    }

    void buildingLod1Mesh::removeNoise(std::vector<Point_3>& oldPointSets, std::vector<Point_3>& newPointSets, double ratio, std::vector<double> pz) {
        std::vector<double> radius;
        Tree tree(oldPointSets.begin(), oldPointSets.end());
        double sumRadius = 0;
        double sum_z = 0;
        for (double pointz : pz) {
            sum_z += pointz;
        }
        for (Point_3 searchPoint : oldPointSets) {
            K_neighbor_search search(tree, searchPoint, 50);
            double distance = 0;
            for (K_neighbor_search::iterator it = search.begin(); it != search.end(); ++it) {
                distance += it->second;
            }
            distance /= 50;
            sumRadius += distance;
            radius.push_back(distance);
        }

        int pointNum = oldPointSets.size();
        double averageRadius = sumRadius / pointNum;
        double averageZ = sum_z / pointNum;
        double scalRate = (height_z - lowest_z) / averageRadius;
        for (int index = 0; index < pointNum; index++) {
            if ((radius[index] + 0.5 * (pz[index] - lowest_z) / scalRate) < ratio * (averageRadius + 0.5 * (averageZ - lowest_z) / scalRate))
                newPointSets.push_back(oldPointSets[index]);
        }
    }

    void buildingLod1Mesh::getAlphaShape(const std::vector<Point_2>& pointSet, std::vector<Point_2>& contour, int para) {
        Alpha_shape_2 A(pointSet.begin(), pointSet.end(), FT(para), Alpha_shape_2::GENERAL);
        std::vector<Segment> segments;
        alphaEdges(A, std::back_inserter(segments));
        std::map<Point_2, int> flagp;
        int num = 0;
        for (auto& seg : segments) {
            Point_2 p1(seg.vertex(0).x(), seg.vertex(0).y());
            Point_2 p2(seg.vertex(1).x(), seg.vertex(1).y());
            if (flagp[p1] == 0) {
                flagp[p1] = ++num;
            }
            if (flagp[p2] == 0) {
                flagp[p2] = ++num;
            }
        }
        std::vector<std::vector< Point_2>> allcontour;
        while (segments.size() != 0) {
            std::vector<Point_2> currentcontour;
            int headid = flagp[segments.begin()->vertex(1)];
            int endid = flagp[segments.begin()->vertex(0)];
            currentcontour.push_back(segments.begin()->vertex(0));
            segments.erase(segments.begin());
            for (auto iter = segments.begin(); iter != segments.end(); ) {
                if (endid == flagp[iter->vertex(0)]) {
                    break;
                }
                if (headid == flagp[iter->vertex(0)]) {
                    currentcontour.push_back(iter->vertex(0));
                    headid = flagp[iter->vertex(1)];
                    segments.erase(iter);
                    iter = segments.begin();
                    continue;
                }
                else {
                    iter++;
                }
            }
            allcontour.push_back(currentcontour);
        }
        int maxnum = 0;
        int id_max = 0;
        for (int j = 0; j < allcontour.size(); j++) {
            if (allcontour[j].size() > maxnum) {
                id_max = j;
                maxnum = allcontour[j].size();
            }
        }

        for (int j = 0; j < allcontour[id_max].size(); j++) {
            contour.push_back(allcontour[id_max][j]);
        }
    }

    void buildingLod1Mesh::simplify(const std::vector<Point_2>& contour, std::vector<Point_2>& simplified_contour) {
        std::vector<double> Origin_border, Result_border;
        for (auto iter1 = contour.begin(); iter1 != contour.end(); iter1++) {
            Origin_border.push_back(iter1->x());
            Origin_border.push_back(iter1->y());
        }
        psimpl::simplify_douglas_peucker<2>(Origin_border.begin(), Origin_border.end(), 3, std::back_inserter(Result_border));

        for (int i = 0; i < Result_border.size() / 2; ++i) {
            Point_2 currentpoint(Result_border[2 * i + 0], Result_border[2 * i + 1]);
            simplified_contour.push_back(currentpoint);
        }
    }

    void buildingLod1Mesh::buildMesh(Mesh& mesh)
    {
        //插入顶点并保存句柄
        std::vector<vertex_descriptor> upVertices;
        std::vector<vertex_descriptor> downVertices;
        for (int i = 0;i<simplfyPoints.size()-1;i++)
        {
            vertex_descriptor upVertex = mesh.add_vertex(Point_3(simplfyPoints[i].x(), simplfyPoints[i].y(), height_z));
            vertex_descriptor downVertex = mesh.add_vertex(Point_3(simplfyPoints[i].x(), simplfyPoints[i].y(), lowest_z));
            upVertices.push_back(upVertex);
            downVertices.push_back(downVertex);
        }
        //添加顶部
        reverse(upVertices.begin(), upVertices.end());
        reverse(downVertices.begin(), downVertices.end());

        mesh.add_face(upVertices);
        //底部顶点反向
        std::vector<vertex_descriptor>downVertices1;
        for (int i = downVertices.size() - 1; i >= 0; i--)
        {
            downVertices1.push_back(downVertices[i]);
        }
        //添加底部
        mesh.add_face(downVertices1);
        //顶点链闭环
        vertex_descriptor upTail = upVertices[0];
        upVertices.push_back(upTail);
        vertex_descriptor downTail = downVertices[0];
        downVertices.push_back(downTail);

        //添加侧面
        for (int index = 0; index < upVertices.size() - 1; index++)
        {
            mesh.add_face(upVertices[index + 1], upVertices[index], downVertices[index], downVertices[index + 1]);
        }
        PMP::triangulate_faces(mesh);

    }

    pointcloudProjection::pointcloudProjection(std::map<int, std::vector<Point_3>> pointsets)
        : pointSets(pointsets) {}

    void pointcloudProjection::projection(std::map<int, std::vector<Point_3>>& newPointSets, int neightborPoint, int groudPointTag) {
        if (neightborPoint <= 0) {
            std::cout << "neightborPoint number error with value " << neightborPoint << std::endl;
            return;
        }
        Tree tree(pointSets[groudPointTag].begin(), pointSets[0].end());
        for (auto& point_set : pointSets) {
            if (point_set.first == groudPointTag) {
                newPointSets[groudPointTag] = point_set.second;
                continue;
            }

            std::vector<Point_3> newpoints;
            for (Point_3 searchPoint : point_set.second) {
                K_neighbor_search search(tree, searchPoint, neightborPoint);
                double height = 0;
                for (K_neighbor_search::iterator it = search.begin(); it != search.end(); ++it) {
                    height += it->first.hz();
                }
                height /= neightborPoint;
                Point_3 newpoint(searchPoint.hx(), searchPoint.hy(), height);
                newpoints.push_back(newpoint);
            }
            newPointSets[point_set.first] = newpoints;
        }
    }
}
