file(
	GLOB SRC_PPO *.cc *.h
)
list(SORT SRC_PPO)

set(SRC_PPO ${SRC_PPO} PARENT_SCOPE)
