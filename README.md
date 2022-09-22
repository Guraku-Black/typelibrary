#typelibrary
/*list of basic library files

1. Turn off optimization for bmp file error.
  ⦁	under "Properties" - "C/C++" - "Optimization", Select Disabled.
  
2. To add DLL support. 
  Make sure to select all Configuration. Both Debug and release.
  
  a. under Properties > C/C++ > General, Select "Additional Include Directories ", add dll directory.
  b. under Properties > Linker > Input, select "Additional Dependencies" add the name of the library, Example "cudalibrary.lib".
  c. under Properties > Linker > General, select "Additional Library Directories", add directory path to the library.
  d. Add Post Build Event. 
     under Properties > Build Events > Post-Build Event. 
     add -->  xcopy /y /d "E:\Projects\Visual Studio\cudalibrary\x64\Release\cudalibrary.dll" "$(OutDir)"
     Modify string to include path to dll.



