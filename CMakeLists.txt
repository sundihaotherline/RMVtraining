cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
option(BUILD_TESTING "Build the starter-project tests" ON)
option(ENABLE_MVS_CAMERA "Build with HIKROBOT MVS camera capture support" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d imgcodecs highgui)
find_package(Threads REQUIRED)

# 海康 MVS 相机 SDK（可选）。找不到时仍可编译，get_pic 退化为离线图片源。
set(HAS_MVS_SDK FALSE)
if(ENABLE_MVS_CAMERA)
    find_path(MVS_INCLUDE_DIR NAMES MvCameraControl.h
              PATHS /opt/MVS/include NO_DEFAULT_PATH)
    find_library(MVS_LIBRARY NAMES libMvCameraControl.so MvCameraControl
                 PATHS /opt/MVS/lib/64 /opt/MVS/lib/32 NO_DEFAULT_PATH)
    if(MVS_INCLUDE_DIR AND MVS_LIBRARY)
        message(STATUS "MVS SDK found: ${MVS_INCLUDE_DIR}")
        set(HAS_MVS_SDK TRUE)
    else()
        message(STATUS "MVS SDK not found; camera capture falls back to image source")
    endif()
endif()

add_subdirectory(serialPort)
if(ENABLE_WEB_DEBUG)
    add_subdirectory(web_debug)
endif()

add_library(algorithm_support STATIC
    armor/src/BulletModel.cpp
    armor/src/KalmanFilter.cpp
)
target_include_directories(algorithm_support PUBLIC armor/include)
target_link_libraries(algorithm_support PUBLIC Eigen3::Eigen)

add_library(training_core STATIC
    src/AppSupport.cpp
    src/AutoAimTypes.cpp
    student/src/StudentTasks.cpp
    student/src/ArmorDetector.cpp
    student/src/NumberClassifier.cpp
    student/src/CameraStream.cpp
)
target_include_directories(training_core
    PUBLIC
        include
        student/include
        serialPort/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(training_core PUBLIC serialPort ${OpenCV_LIBS})
if(HAS_MVS_SDK)
    target_include_directories(training_core PRIVATE ${MVS_INCLUDE_DIR})
    target_compile_definitions(training_core PRIVATE HAS_MVS_SDK)
    target_link_libraries(training_core PUBLIC ${MVS_LIBRARY})
endif()

add_executable(rmv_training main.cpp)
target_include_directories(rmv_training PRIVATE web_debug/include)
target_link_libraries(rmv_training
    PRIVATE
        training_core
        algorithm_support
        Threads::Threads
)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(rmv_training PRIVATE web_debug)
endif()
if(HAS_MVS_SDK)
    # 运行期无需额外设置 LD_LIBRARY_PATH。
    set_property(TARGET rmv_training APPEND PROPERTY BUILD_RPATH "/opt/MVS/lib/64")
endif()

if(BUILD_TESTING)
    enable_testing()
    add_executable(test_scaffold tests/test_scaffold.cpp)
    target_link_libraries(test_scaffold PRIVATE training_core)
    add_test(NAME scaffold_safety COMMAND test_scaffold)
    add_test(NAME cli_help COMMAND rmv_training --help)
endif()
