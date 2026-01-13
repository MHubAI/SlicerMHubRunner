cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MHubRunner)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://mhub.ai/")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Leonard Nürnberg (AIM, Mass General Brigham)")
set(EXTENSION_DESCRIPTION "MHub Runner is a 3D Slicer plugin that seamlessly integrates Deep Learning models from the Medical Hub repository (MHub) into 3D Slicer.")
set(EXTENSION_ICONURL "https://mhub.ai/slicer/SlicerMHubRunner.png")
set(EXTENSION_SCREENSHOTURLS "https://mhub.ai/slicer/image01.png")
set(EXTENSION_DEPENDS "QuantitativeReporting") # list of dependencies, space separated, e.g., '"d1" "d2" ...'

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MHubRunner)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
