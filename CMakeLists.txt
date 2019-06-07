cmake_minimum_required(VERSION 2.8.3)
project(relevance_map)

file(GLOB_RECURSE INCLUDE_FILES include/${PROJECT_NAME}/*)
file(GLOB_RECURSE SRC_FILES src/relevance_map_node.cpp include/${PROJECT_NAME}/*)
file(GLOB_RECURSE RGBD_SRC_FILES src/rgbd_utils/*)
file(GLOB_RECURSE RGBD_INCLUDE_FILES include/rgbd_utils/*)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC")

set(CATKIN_COMMON_COMPONENTS roscpp
                             rospy
                             std_msgs
                             message_generation
                             image_transport
                             image_geometry
                             cv_bridge
                             pcl_conversions
                             message_filters
                             geometry_msgs
                             sensor_msgs
                             depth_image_proc
                            )

find_package(catkin REQUIRED COMPONENTS ${CATKIN_COMMON_COMPONENTS})
find_package(OpenCV REQUIRED) # FIXME nice to tell a version range.
find_package(PCL REQUIRED)
find_library(BASE64 NAMES b64)



SET(CMM_INSTALL_TREE FALSE CACHE STRING "Path to CMM.")

if(NOT CMM_INSTALL_TREE)
  message(SEND_ERROR "Please tell me where to find a compiled CMM.  For example:
      cmake -DCMM_INSTALL_TREE=/path/to/cmm.installtree/ .
Then you can run again.")
endif(NOT CMM_INSTALL_TREE)

add_library(cmm SHARED IMPORTED)
set_property(TARGET cmm PROPERTY IMPORTED_LOCATION "${CMM_INSTALL_TREE}/lib/libcmm.so" )
# The /lib/libcmm.so above could be further cleaned up.



SET(IMAGE_PROCESSING_INSTALL_TREE FALSE CACHE STRING "Path to image_processing.")

if(NOT IMAGE_PROCESSING_INSTALL_TREE)
  message(SEND_ERROR "Please tell me where to find a compiled IMAGE_PROCESSING.  For example:
      cmake -DIMAGE_PROCESSING_INSTALL_TREE=/path/to/image_processing.installtree/ .
Then you can run again.")
endif(NOT IMAGE_PROCESSING_INSTALL_TREE)

add_library(image_processing SHARED IMPORTED)
set_property(TARGET image_processing PROPERTY IMPORTED_LOCATION "${IMAGE_PROCESSING_INSTALL_TREE}/lib/libimage_processing.so" )
# The /lib/libimage_processing.so above could be further cleaned up.



include_directories(include
	${PCL_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
        ${OpencCV_INCLUDE_DIRS}
BEFORE SYSTEM        ${IMAGE_PROCESSING_INSTALL_TREE}/include "${CMM_INSTALL_TREE}/include"
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})





# # find_library(CMM_LIBRARY cmm)
# find_package(cmm)

include_directories(${PCL_INCLUDE_DIRS} /usr/local/include include ${OpenCV_INCLUDE_DIRS} "${CMM_INSTALL_TREE}/include" )
link_directories(${PCL_LIBRARY_DIRS} /usr/local/lib)
add_definitions(${PCL_DEFINITIONS})




## Generate messages in the 'msg' folder
#add_message_files(
#        FILES
#        rgbd_motion_data.msg
#        joints_values.msg
#        motion_rect.msg
#        dataset.msg
#        sv_feature.msg
#        gmm_archive.msg
#        endeffector_pose.msg
#        sync_dataset.msg
#        target_info.msg
#)

## Generate services in the 'srv' folder
add_service_files(
        FILES
        cnn_features.srv
)

## Generate actions in the 'action' folder
#add_action_files(
#        FILES
#        pose_goal.action
#)

## Generate added messages and services with any dependencies listed here
generate_messages(
        DEPENDENCIES
        std_msgs
        sensor_msgs
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES relevance_map
        CATKIN_DEPENDS roscpp rospy std_msgs message_runtime gazebo_msgs sensor_msgs
        DEPENDS ${IMAGE_PROCESSING_LIBRARY}
        #  DEPENDS system_lib
)



add_library(rgbd_utils ${RGBD_SRC_FILES})
target_link_libraries(rgbd_utils ${catkin_LIBRARIES})

install(TARGETS rgbd_utils
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(DIRECTORY include/rgbd_utils
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/..)

add_library(${PROJECT_NAME} ${SRC_FILES})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES}  cmm image_processing ${IMAGE_PROCESSING_LIBRARY} rgbd_utils)


## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})



## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})



#TOOLS
add_executable(pcd_to_relevance_map tools/pcd_to_relevance_map.cpp)
target_link_libraries(pcd_to_relevance_map ${catkin_LIBRARIES} relevance_map cmm image_processing ${IMAGE_PROCESSING_LIBRARY} ${PCL_LIBRARIES} tbb)

#add_executable(evaluate_classifier tools/evaluate_classifier.cpp)
#target_link_libraries(evaluate_classifier ${catkin_LIBRARIES} relevance_map cmm image_processing ${image_processing_LIBRARIES} ${PCL_LIBRARIES} tbb yaml-cpp.so)

message(WARNING "FIXME remove IMAGE_PROCESSING_LIBRARY=${IMAGE_PROCESSING_LIBRARY}")


add_executable(evaluate_archive tools/evaluate_archive.cpp)
target_link_libraries(evaluate_archive ${catkin_LIBRARIES} relevance_map cmm image_processing ${IMAGE_PROCESSING_LIBRARY} ${PCL_LIBRARIES} tbb yaml-cpp.so)

add_executable(features_variability tools/features_variability.cpp)
target_link_libraries(features_variability ${catkin_LIBRARIES} relevance_map rgbd_utils cmm image_processing ${IMAGE_PROCESSING_LIBRARY} ${PCL_LIBRARIES} tbb yaml-cpp.so)

#TEST
add_executable(test_rgbd_utils src/rgbd_utils/test_rgbd_utils.cpp)
target_link_libraries(test_rgbd_utils ${catkin_LIBRARIES} rgbd_utils)

#EXAMPLE
add_executable(example_node examples/example_node.cpp)
target_link_libraries(example_node ${catkin_LIBRARIES} relevance_map rgbd_utils cmm ${PCL_LIBRARIES}  image_processing tbb yaml-cpp.so)
