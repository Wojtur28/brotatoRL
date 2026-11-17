extends Node

# Lab probe: answers the open questions of the plan (does the mod load, where user:// is,
# Steam state, window/render flags, simulation speed), writes a JSON report and quits.
# Options come from --probe-<key>=<value> command line arguments.

const MOD_DIR := "res://mods-unpacked/BrotatoRL-Probe/"
const REPORT_FILE := "user://rl_probe_report.json"
const TITLE_SCENE := "res://ui/menus/title_screen/title_screen.tscn"
const GAME_SCENE := "res://main.tscn"

var _args := {}
var _report := {}
var _boot_usec := 0
var _run_started := false
var _finished := false
var _measure_start_frame := -1
var _measure_start_usec := 0


func _init() -> void:
	_boot_usec = OS.get_ticks_usec()
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--probe-"):
			var parts: PoolStringArray = arg.trim_prefix("--probe-").split("=", true, 1)
			_args[parts[0]] = parts[1] if parts.size() > 1 else "1"

	_report["mode"] = _args.get("mode", "info")
	_report["user_dir_at_mod_init"] = OS.get_user_data_dir()
	_report["cmdline"] = Array(OS.get_cmdline_args())
	_report["executable"] = OS.get_executable_path()
	_report["engine"] = Engine.get_version_info()
	_report["platform_type"] = Platform.get_type_as_string()

	var impl = Platform._platform_impl
	if "init_error" in impl:
		_report["steam_init_error_before_mod"] = impl.init_error
		_report["steam_id"] = str(impl.steam_id)
		# Block achievements and stat writes for the rest of the session.
		impl.init_error = true

	pause_mode = PAUSE_MODE_PROCESS
	ModLoaderMod.install_script_extension(MOD_DIR + "extensions/singletons/announcement_manager.gd")


func _ready() -> void:
	# This build has no --disable-vsync flag; vsync would cap the unthrottled loop.
	OS.vsync_enabled = false
	DebugService.disable_saving = true
	DebugService.generate_full_unlocked_save_file = true
	var _e = ProgressData.connect("ready", self, "_on_progress_data_ready")


func _on_progress_data_ready() -> void:
	var settings: Dictionary = ProgressData.settings
	settings.fullscreen = false
	settings.pause_on_focus_lost = false
	settings.mute_on_focus_lost = false
	settings.limit_fps = false
	settings.volume = {"master": 0.0, "sound": 0.0, "music": 0.0}

	_report["user_dir_after_progress_data"] = OS.get_user_data_dir()
	_report["save_dir"] = ProgressData.SAVE_DIR
	_report["characters_unlocked"] = ProgressData.characters_unlocked.size()
	_report["items_unlocked"] = ProgressData.items_unlocked.size()


func _process(_delta: float) -> void:
	if _run_started or _finished:
		return
	var scene = get_tree().current_scene
	if scene == null or scene.filename != TITLE_SCENE:
		return

	_report["title_reached_idle_frames"] = Engine.get_idle_frames()
	_report["title_reached_wall_ms"] = (OS.get_ticks_usec() - _boot_usec) / 1000.0
	if _report.mode == "speed":
		_start_run()
	else:
		_finish()


func _start_run() -> void:
	_run_started = true
	VisualServer.render_loop_enabled = _args.get("render", "1") == "1"

	ProgressData.reset_and_save_new_run_state()
	RunData.reset(false)

	var character = ItemService.get_element_safe(ItemService.characters, _args.get("character", "character_well_rounded"))
	RunData.add_character(character, 0)
	if RunData.some_player_has_weapon_slots():
		var choices: Array = ItemService.get_ordered_starting_weapons(character.starting_weapons)
		var _weapon = RunData.add_weapon(choices[0], 0, true)
	RunData.add_starting_items_and_weapons()

	var difficulty = null
	for diff in ItemService.difficulties:
		if diff.value == int(_args.get("danger", "0")):
			difficulty = diff
	RunData.current_difficulty = difficulty.value
	RunData.reset_elites_spawn()
	RunData.init_elites_spawn()
	RunData.init_events_nightmare()
	RunData.enabled_dlcs = ProgressData.get_active_dlc_ids()
	for effect in difficulty.effects:
		effect.apply(0)
	for player_data in RunData.players_data:
		player_data.uses_ban = RunData.is_ban_mode_active
		player_data.remaining_ban_token = RunData.BAN_MAX_TOKEN
	RunData.init_bosses_spawn()
	RunData.current_run_accessibility_settings = ProgressData.settings.enemy_scaling.duplicate()
	ProgressData.load_status = LoadStatus.SAVE_OK

	RunData.current_wave = int(_args.get("wave", "1"))
	RunData.invulnerable = true
	_report["character"] = character.my_id
	_report["weapons"] = []
	for weapon in RunData.get_player_weapons(0):
		_report.weapons.push_back(weapon.my_id)
	var _e = get_tree().change_scene(GAME_SCENE)


func _physics_process(_delta: float) -> void:
	if not _run_started or _finished:
		return
	var scene = get_tree().current_scene
	if scene == null or scene.filename != GAME_SCENE:
		return

	var frame := Engine.get_physics_frames()
	var frames := int(_args.get("frames", "900"))
	if _measure_start_frame < 0:
		_measure_start_frame = frame + 120
	elif frame == _measure_start_frame:
		_measure_start_usec = OS.get_ticks_usec()
	elif frame == _measure_start_frame + frames:
		var elapsed := (OS.get_ticks_usec() - _measure_start_usec) / 1000000.0
		_report["measured_frames"] = frames
		_report["wall_seconds"] = elapsed
		_report["physics_fps"] = frames / elapsed
		_report["enemies"] = scene._entity_spawner.enemies.size()
		_report["enemy_projectiles"] = scene._enemy_projectiles.get_child_count() if is_instance_valid(scene._enemy_projectiles) else -1
		_report["wave"] = RunData.current_wave
		_report["wave_time_left"] = scene._wave_timer.time_left
		_report["player_hp"] = RunData.get_player_current_health(0)
		_finish()


func _finish() -> void:
	_finished = true
	_report["render_loop_enabled"] = VisualServer.render_loop_enabled
	_report["window_size"] = [OS.window_size.x, OS.window_size.y]
	_report["fullscreen"] = OS.window_fullscreen
	_report["vsync"] = OS.vsync_enabled
	_report["target_fps"] = Engine.target_fps
	_report["iterations_per_second"] = Engine.iterations_per_second
	_report["static_memory_mb"] = OS.get_static_memory_usage() / 1048576.0
	_report["user_dir_at_finish"] = OS.get_user_data_dir()

	var file := File.new()
	if file.open(REPORT_FILE, File.WRITE) == OK:
		file.store_string(to_json(_report))
		file.close()
	print("[BrotatoRL probe] " + to_json(_report))
	get_tree().quit()
