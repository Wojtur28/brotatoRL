# BrotatoRL lab shim. override.cfg (lab copy only) loads this script instead of the
# original ModLoaderStore autoload. It makes the Steam build load mod zips from the
# local mods/ folder, the same way the GOG and Epic builds do.
extends "res://addons/mod_loader/mod_loader_store.gd"

const EXPECTED_USER_DIR_NAME := "BrotatoRL"


func _init() -> void:
	# Safety net: never run against the real Brotato user folder.
	var user_dir := OS.get_user_data_dir()
	if user_dir.get_file() != EXPECTED_USER_DIR_NAME:
		printerr("[BrotatoRL shim] unexpected user dir %s, quitting" % user_dir)
		OS.kill(OS.get_process_id())
		return

	ml_options.steam_workshop_enabled = false
	ml_options.enable_mods = true
	print("[BrotatoRL shim] local mods enabled, user dir: %s" % user_dir)
