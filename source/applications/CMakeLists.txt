set (APPLICATION_NAME imagetransferstyle)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
	"imagetransferstyle/main.h.in"
	"${PROJECT_SOURCE_DIR}/source/applications/imagetransferstyle/main.h"
)

add_subdirectory(imagetransferstyle)
