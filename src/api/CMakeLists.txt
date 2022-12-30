file(
	GLOB SRC_API *.cc *.h
)
list(SORT SRC_API)

set(SRC_API ${SRC_API} PARENT_SCOPE)
