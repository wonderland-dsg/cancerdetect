
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}" )

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Find the QtWidgets library
find_package(Qt5Widgets REQUIRED)


#find OpenMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set (FOO_MOC_HEADERS
        mainwindow.h)


find_package(OpenCV REQUIRED)

set(SOURCE_FILES
        main.cpp
        svm.h
        svm.cpp
        LBP.h
        LBP.cpp
        breastcancer_predict.h
        breastcancer_predict.cpp
        GLCM.h
        GLCM.cpp
        predict_glcm.h predict_glcm.cpp normData.h normData.cpp my_scale.h mysvmtrain.h mysvmtrain.cpp
        mainwindow.h mainwindow.cpp
        cvmatandqimage.h cvmatandqimage.cpp Canny.h)

add_executable(breast_concer_detection ${SOURCE_FILES})

target_link_libraries(breast_concer_detection Qt5::Widgets ${OpenCV_LIBS} libtbb.so.2 )
#${Qt5Widgets_LIBRARIES}