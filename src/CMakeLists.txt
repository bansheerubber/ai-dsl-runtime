file(
	GLOB SRC_DSL *.cc *.h
)
list(SORT SRC_DSL)

set(SRC_DSL ${SRC_DSL} PARENT_SCOPE)

add_subdirectory(api)
set(SRC_API ${SRC_API} PARENT_SCOPE)
