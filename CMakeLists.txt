cmake_minimum_required(VERSION 2.8.3)
project(relevance_map)

file(GLOB_RECURSE INCLUDE_FILES include/${PROJECT_NAME}/*)
file(GLOB_RECURSE SRC_FILES src/* include/${PROJECT_NAME}/*)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC")

set(CATKIN_COMMON_COMPONENTS roscpp
                             rospy
                             std_msgs
                             message_generation
                             image_transport
                             image_geometry
                             rgbd_utils
                             message_filters
                             geometry_msgs
                             sensor_msgs
                            )

find_package(catkin REQUIRED COMPONENTS ${CATKIN_COMMON_COMPONENTS})
find_package(OpenCV 2.4.8 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_library(BASE64 NAMES b64)


include_directories(include
	${PCL_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
        ${OpencCV_INCLUDE_DIRS}
        ${RGBD_UTILS_INCLUDE_DIR}
BEFORE SYSTEM        ${image_processing_INCLUDE} ${IAGMM_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(LOCAL_LIB_DIR /usr/local/ CACHE STRING "local libraries directory for iagmm and image_processing")
set(IAGMM_INCLUDE_DIR ${LOCAL_LIB_DIR}include)
set(IAGMM_LIBRARIES ${LOCAL_LIB_DIR}lib/libiagmm.so)
set(image_processing_INCLUDE_DIR ${LOCAL_LIB_DIR}include/)
set(image_processing_LIBRARIES ${LOCAL_LIB_DIR}lib/libimage_processing.so)

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
#add_service_files(
#        FILES
#        is_moving.srv
#)

## Generate actions in the 'action' folder
#add_action_files(
#        FILES
#        pose_goal.action
#)

## Generate added messages and services with any dependencies listed here
#generate_messages(
#        DEPENDENCIES
#        std_msgs
#        sensor_msgs
#        actionlib_msgs
#        geometry_msgs
#)

catkin_package(
        INCLUDE_DIRS include/${PROJECT_NAME}/
        LIBRARIES relevance_map
        CATKIN_DEPENDS roscpp rospy std_msgs message_runtime rgbd_utils gazebo_msgs sensor_msgs
        DEPENDS ${IMAGE_PROCESSING_LIBRARY}
        #  DEPENDS system_lib
)


add_library(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${IAGMM_LIBRARIES} ${image_processing_LIBRARIES})

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})


## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})


