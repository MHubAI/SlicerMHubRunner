#-----------------------------------------------------------------------------
set(MODULE_NAME MHubRunner)

#-----------------------------------------------------------------------------
# Build information
set(MHUBRUNNER_BUILD_REVISION "unknown")
set(MHUBRUNNER_BUILD_REVISION_SHORT "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    RESULT_VARIABLE _mhubrunner_git_result
    OUTPUT_VARIABLE _mhubrunner_git_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    )
  if(_mhubrunner_git_result EQUAL 0 AND _mhubrunner_git_revision)
    set(MHUBRUNNER_BUILD_REVISION "${_mhubrunner_git_revision}")
    string(SUBSTRING "${MHUBRUNNER_BUILD_REVISION}" 0 12 MHUBRUNNER_BUILD_REVISION_SHORT)
  endif()
endif()

# Generate metadata inside the internal package so Slicer does not discover it as a module.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/MHubRunnerLib")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/MHubRunnerLib/build_info.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/MHubRunnerLib/build_info.py"
  @ONLY
  )

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  MHubRunnerLib/__init__.py
  MHubRunnerLib/gpu_requirements.py
  ${CMAKE_CURRENT_BINARY_DIR}/MHubRunnerLib/build_info.py
  MHubRunnerModelHandlers/__init__.py
  MHubRunnerModelHandlers/base.py
  MHubRunnerModelHandlers/handlers/__init__.py
  MHubRunnerModelHandlers/handlers/gc_grt123_lung_cancer.py
  MHubRunnerModelHandlers/registry.py
  MHubRunnerModelHandlers/run_manifest.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/MHubRunner_icon_b.png
  Resources/Icons/MHubRunner_icon_w.png
  Resources/Icons/MRunner_b.png
  Resources/Icons/MRunner_w.png
  Resources/Icons/Name.png
  Resources/Icons/hi_cancel_b.png
  Resources/Icons/hi_cancel_w.png
  Resources/Icons/hi_info_b.png
  Resources/Icons/hi_info_w.png
  Resources/Icons/hi_modelcard_b.png
  Resources/Icons/hi_modelcard_w.png
  Resources/Icons/hi_noplay_b.png
  Resources/Icons/hi_noplay_w.png
  Resources/Icons/hi_play_b.png
  Resources/Icons/hi_play_w.png
  Resources/Icons/hi_pull_b.png
  Resources/Icons/hi_pull_w.png
  Resources/Icons/hi_pulled_b.png
  Resources/Icons/hi_pulled_w.png
  Resources/Icons/hi_settings_b.png
  Resources/Icons/hi_settings_w.png
  Resources/Icons/hi_show_b.png
  Resources/Icons/hi_show_w.png
  Resources/Icons/hi_noshow_b.png
  Resources/Icons/hi_noshow_w.png
  Resources/Icons/hi_running_b.png
  Resources/Icons/hi_running_w.png
  Resources/UI/MHubRunner.ui
  Resources/UI/MHubRunnerSettings.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Register the focused module tests maintained under Testing.
  add_subdirectory(Testing)
endif()
