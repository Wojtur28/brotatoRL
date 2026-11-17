extends Node

# BrotatoRL RLBridge: lets a Python agent drive Brotato runs (protocol in bridge.gd).
# The mod stays inactive unless the game is started with --rl-port=<port>.
# Options: --rl-<key>=<value> (port, render, realtime, lite, sound, unlock_all).

const MOD_ID := "BrotatoRL-RLBridge"
const MOD_DIR := "res://mods-unpacked/BrotatoRL-RLBridge/"


func _init() -> void:
	var options := {}
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--rl-"):
			var parts: PoolStringArray = arg.trim_prefix("--rl-").split("=", true, 1)
			options[parts[0]] = parts[1] if parts.size() > 1 else "1"

	if not options.has("port"):
		ModLoaderLog.info("No --rl-port argument, staying inactive", MOD_ID)
		return

	# Steam may already be initialised (Platform loads before mods). Flagging it as failed
	# turns achievements, stat writes and Workshop queries into no-ops for this session.
	var platform_impl = Platform._platform_impl
	options["steam_connected"] = "init_error" in platform_impl and not platform_impl.init_error
	if "init_error" in platform_impl:
		platform_impl.init_error = true

	ModLoaderMod.install_script_extension(MOD_DIR + "extensions/singletons/announcement_manager.gd")
	ModLoaderMod.install_script_extension(MOD_DIR + "extensions/entities/units/movement_behaviors/player_movement_behavior.gd")

	var bridge: Node = load(MOD_DIR + "bridge.gd").new()
	bridge.name = "Bridge"
	bridge.setup(options)
	add_child(bridge)
	ModLoaderLog.info("Bridge created, port %s" % options.port, MOD_ID)
