#-----------------------------------------------------------------------------
set(MODULE_NAME MHubRunner)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/MHubRunner.png
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

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
