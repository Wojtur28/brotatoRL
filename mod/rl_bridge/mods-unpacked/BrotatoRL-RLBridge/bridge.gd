extends Node

# Lockstep bridge between Brotato and a Python agent over localhost TCP.
#
# Framing: 4-byte little-endian length + UTF-8 JSON. The game sends a message and then
# blocks inside _physics_process until Python replies, so the whole game is frozen while
# Python decides. With --fixed-fps 60 every physics frame is exactly 1/60 s of game time,
# which keeps each observation in sync with the action that follows it.
#
# Game -> Python                         Python -> game (reply)
#   hello     (at the title screen)        reset / quit
#   obs       (every frame_skip frames)    act / reset / quit
#   decision  (level-up, crate, shop)      choose / reset / quit
#   end       (death or victory)           reset / quit

const PROTOCOL_VERSION := 1
const MOD_VERSION := "0.5.0"
const ENEMY_COLS := 10
const BRIDGE_META := "brotato_rl_bridge"
const TITLE_SCENE := "res://ui/menus/title_screen/title_screen.tscn"
const GAME_SCENE := "res://main.tscn"
const SHOP_SCENE := "res://ui/menus/shop/shop.tscn"
const END_RUN_SCENE := "res://ui/menus/run/end_run.tscn"
const FRAME_USEC := 16667
const CONNECT_TIMEOUT_MS := 15000
const SHOP_SETTLE_FRAMES := 3
const MAX_SHOP_ACTIONS := 60
const MAX_REPEATED_DECISIONS := 3
const PLAYER_STATS := [
	["max_hp", "stat_max_hp_hash"],
	["hp_regeneration", "stat_hp_regeneration_hash"],
	["lifesteal", "stat_lifesteal_hash"],
	["percent_damage", "stat_percent_damage_hash"],
	["melee_damage", "stat_melee_damage_hash"],
	["ranged_damage", "stat_ranged_damage_hash"],
	["elemental_damage", "stat_elemental_damage_hash"],
	["attack_speed", "stat_attack_speed_hash"],
	["crit_chance", "stat_crit_chance_hash"],
	["engineering", "stat_engineering_hash"],
	["range", "stat_range_hash"],
	["armor", "stat_armor_hash"],
	["dodge", "stat_dodge_hash"],
	["speed", "stat_speed_hash"],
	["luck", "stat_luck_hash"],
	["harvesting", "stat_harvesting_hash"],
]

const UNLOCK_FIELDS := [
	"zones_unlocked", "characters_unlocked", "upgrades_unlocked", "consumables_unlocked",
	"weapons_unlocked", "items_unlocked", "challenges_completed",
]

enum Phase { BOOT, IDLE, STARTING, BETWEEN, COMBAT }

# Read by the player movement extension.
var controls_player := false
var move := Vector2.ZERO

var _options := {}
var _peer := StreamPeerTCP.new()
var _phase: int = Phase.BOOT
var _frame_skip := 4
var _frames_since_obs := 0
var _step := 0
var _run_start_frame := 0
var _stale_scene_id := 0
var _realtime := false
var _speed := 1.0  # with realtime: game seconds per wall-clock second (2.0 = twice as fast)
var _rt_origin_usec := 0
var _rt_frames := 0
var _stat_hashes := []
var _type_ids := {}
var _new_types := {}
var _hooked_player_id := 0
var _unlocks := {}

var _shop_id := 0
var _shop_frames := 0
var _shop_actions := 0
var _last_decision_key := ""
var _repeated_decisions := 0

var _damage_taken := 0
var _hits_taken := 0
var _dodges := 0
var _healed := 0
var _profile := false
var _obs_done_usec := 0
var _send_usec := 0
var _base_kills := 0
var _base_materials := 0
var _base_trees := 0


func setup(options: Dictionary) -> void:
	_options = options
	_realtime = options.get("realtime", "0") == "1"


func _ready() -> void:
	pause_mode = PAUSE_MODE_PROCESS
	process_priority = -1000000
	Engine.set_meta(BRIDGE_META, self)

	# This build has no --disable-vsync flag, and vsync would cap the unthrottled loop.
	# Rendering is switched off only after the title screen (the boot scene relies on _draw).
	OS.vsync_enabled = false

	# Lab profile: never write saves. Unlocks are applied at the title screen (see _tick_boot).
	DebugService.disable_saving = true
	var _e = ProgressData.connect("ready", self, "_on_progress_data_ready")

	_connect_to_python()


func _on_progress_data_ready() -> void:
	var settings: Dictionary = ProgressData.settings
	settings.fullscreen = false
	settings.pause_on_focus_lost = false
	settings.mute_on_focus_lost = false
	settings.on_lost_focus = 0
	settings.limit_fps = false
	settings.optimize_end_waves = true
	settings.manual_aim = false
	settings.manual_aim_on_mouse_press = false
	settings.mouse_only = false
	settings.endless_mode_toggled = false
	# Ban buttons act immediately instead of a ~40-frame "hold to ban" animation.
	settings.holding_button = false
	if _options.get("sound", "0") != "1":
		settings.volume = {"master": 0.0, "sound": 0.0, "music": 0.0}
	if _options.get("lite", "0") == "1":
		settings.screenshake = false
		settings.visual_effects = false
		settings.damage_display = false


# --- Connection -------------------------------------------------------------------

func _connect_to_python() -> void:
	var port := int(_options.get("port", "0"))
	if _peer.connect_to_host("127.0.0.1", port) != OK:
		printerr("[RLBridge] cannot connect to port %d" % port)
		get_tree().quit()
		return
	var deadline := OS.get_ticks_msec() + CONNECT_TIMEOUT_MS
	while _peer.get_status() == StreamPeerTCP.STATUS_CONNECTING and OS.get_ticks_msec() < deadline:
		OS.delay_msec(5)
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		printerr("[RLBridge] connection to port %d failed" % port)
		get_tree().quit()
		return
	_peer.set_no_delay(true)
	print("[RLBridge] connected to 127.0.0.1:%d" % port)


func _send(msg: Dictionary) -> void:
	var body: PoolByteArray = to_json(msg).to_utf8()
	_peer.put_u32(body.size())
	var _err = _peer.put_data(body)


# Blocks until Python replies. Returns {} when the connection is gone.
func _recv() -> Dictionary:
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return {}
	var size: int = _peer.get_u32()
	if size <= 0 or _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return {}
	var chunk: Array = _peer.get_data(size)
	if chunk[0] != OK:
		return {}
	var parsed := JSON.parse(chunk[1].get_string_from_utf8())
	if parsed.error != OK or typeof(parsed.result) != TYPE_DICTIONARY:
		printerr("[RLBridge] bad message from Python")
		return {}
	return parsed.result


func _shutdown() -> void:
	controls_player = false
	_phase = Phase.IDLE
	_peer.disconnect_from_host()
	get_tree().quit()


# Handles the replies every request accepts. Returns true when the reply was consumed.
func _handle_control(reply: Dictionary) -> bool:
	match reply.get("type", ""):
		"reset":
			_start_run(reply)
			return true
		"quit", "":
			_shutdown()
			return true
	return false


func _await_command() -> void:
	while true:
		var reply := _recv()
		if _handle_control(reply):
			return
		printerr("[RLBridge] expected reset or quit, got %s" % reply.get("type"))


# --- Main loop ----------------------------------------------------------------------

func _physics_process(_delta: float) -> void:
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		if _phase != Phase.IDLE:
			_shutdown()
		return
	if _phase == Phase.IDLE:
		return
	if _realtime and _phase != Phase.BOOT:
		_pace()

	match _phase:
		Phase.BOOT:
			_tick_boot()
		Phase.STARTING, Phase.BETWEEN:
			_tick_between()
		Phase.COMBAT:
			_tick_combat()


func _pace() -> void:
	_rt_frames += 1
	var target := _rt_origin_usec + int(_rt_frames * FRAME_USEC / _speed)
	var now := OS.get_ticks_usec()
	if target > now:
		OS.delay_usec(target - now)
	elif now - target > 250000:
		_rt_origin_usec = now
		_rt_frames = 0


func _tick_boot() -> void:
	var scene = get_tree().current_scene
	if scene == null or scene.filename != TITLE_SCENE:
		return
	_phase = Phase.IDLE
	VisualServer.render_loop_enabled = _options.get("render", "1") == "1"
	# The game's own "full unlocked save" debug path runs while ProgressData boots, before
	# ChallengeService is ready, and makes the game crash at the end of the first wave.
	# Unlocking here, once every singleton is ready, is safe.
	if _options.get("unlock_all", "1") == "1":
		ProgressData.unlock_all()
		ProgressData.set_max_selectable_difficulty()
	_snapshot_unlocks()
	_send(_hello())
	_await_command()


# Challenges completed during a run unlock content in memory. Restoring the snapshot at
# every reset keeps the item pool identical from one episode to the next.
func _snapshot_unlocks() -> void:
	for field in UNLOCK_FIELDS:
		_unlocks[field] = ProgressData.get(field).duplicate()


func _restore_unlocks() -> void:
	for field in UNLOCK_FIELDS:
		ProgressData.set(field, _unlocks[field].duplicate())


func _tick_between() -> void:
	var scene = get_tree().current_scene
	if scene == null or scene.get_instance_id() == _stale_scene_id:
		return

	match scene.filename:
		GAME_SCENE:
			var outcome := _run_outcome(scene)
			if outcome != "":
				_finish_run(scene, outcome)
			elif _is_combat_running(scene):
				_enter_combat(scene)
			elif _phase == Phase.BETWEEN:
				_tick_upgrades(scene)
		SHOP_SCENE:
			_tick_shop(scene)
		END_RUN_SCENE:
			_finish_run(null, "won" if RunData.run_won else "died")


func _tick_combat() -> void:
	var main = get_tree().current_scene
	if main == null or main.filename != GAME_SCENE:
		_leave_combat()
		return
	var outcome := _run_outcome(main)
	if outcome != "":
		_finish_run(main, outcome)
		return
	if main._cleaning_up:
		_leave_combat()
		return
	_frames_since_obs += 1
	if _frames_since_obs >= _frame_skip:
		_send_obs(main)


func _enter_combat(main) -> void:
	_phase = Phase.COMBAT
	_hook_player(main._players[0])
	controls_player = true
	move = Vector2.ZERO
	_send_obs(main)


func _leave_combat() -> void:
	_phase = Phase.BETWEEN
	controls_player = false
	move = Vector2.ZERO


func _send_obs(main) -> void:
	_frames_since_obs = 0
	_step += 1
	var start := OS.get_ticks_usec()
	var msg := _build_state(main)
	msg["type"] = "obs"
	# Profiling (reset with "profile": true), in microseconds: game frames since the previous
	# reply, building this state, sending (JSON + socket) the previous one.
	if _profile:
		msg["perf"] = [start - _obs_done_usec if _obs_done_usec > 0 else 0, OS.get_ticks_usec() - start, _send_usec]
	var sent := OS.get_ticks_usec()
	_send(msg)
	_send_usec = OS.get_ticks_usec() - sent

	var reply := _recv()
	_obs_done_usec = OS.get_ticks_usec()
	if _handle_control(reply):
		return
	if reply.get("type") == "act":
		var m = reply.get("move", [0, 0])
		move = Vector2(float(m[0]), float(m[1])).clamped(1.0)
	else:
		printerr("[RLBridge] expected act, got %s" % reply.get("type"))


func _finish_run(main, outcome: String) -> void:
	_phase = Phase.IDLE
	controls_player = false
	move = Vector2.ZERO
	var msg := _build_state(main)
	msg["type"] = "end"
	msg["outcome"] = outcome
	_send(msg)
	_await_command()


func _is_combat_running(main) -> bool:
	if main._cleaning_up or main._players.empty():
		return false
	var player = main._players[0]
	return is_instance_valid(player) and not player.dead and not main._wave_timer.is_stopped()


func _run_outcome(main) -> String:
	if main._is_run_won:
		return "won"
	if main._is_run_lost:
		return "died"
	if not main._players.empty() and is_instance_valid(main._players[0]) and main._players[0].dead:
		return "died"
	return ""


# --- Starting a run (same steps as character -> weapon -> danger menus) ----------------

func _start_run(msg: Dictionary) -> void:
	var current = get_tree().current_scene
	_stale_scene_id = current.get_instance_id() if current != null else 0
	get_tree().paused = false
	controls_player = false
	move = Vector2.ZERO

	_frame_skip = int(max(1, int(msg.get("frame_skip", 4))))
	_profile = bool(msg.get("profile", false))
	_obs_done_usec = 0
	_send_usec = 0
	if msg.has("render"):
		VisualServer.render_loop_enabled = bool(msg.render)
	if msg.has("realtime"):
		_realtime = bool(msg.realtime)
	if msg.get("speed") != null:
		_speed = clamp(float(msg.speed), 0.05, 100.0)
	_rt_origin_usec = OS.get_ticks_usec()
	_rt_frames = 0
	if msg.get("seed") != null:
		seed(int(msg.seed))
		Utils.set_rng_seed(int(msg.seed))

	var error := _setup_run(msg)
	if error != "":
		_phase = Phase.IDLE
		_send({"type": "error", "message": error})
		_await_command()
		return

	_step = 0
	_frames_since_obs = 0
	_run_start_frame = Engine.get_physics_frames()
	_damage_taken = 0
	_hits_taken = 0
	_dodges = 0
	_healed = 0
	_base_kills = ProgressData.data.enemies_killed
	_base_materials = ProgressData.data.materials_collected
	_base_trees = ProgressData.data.trees_killed
	_phase = Phase.STARTING
	var _e = get_tree().change_scene(GAME_SCENE)


func _setup_run(msg: Dictionary) -> String:
	var character_id := str(msg.get("character", "character_well_rounded"))
	var character = ItemService.get_element_safe(ItemService.characters, character_id)
	if character == null:
		return "unknown character: %s" % character_id

	var danger := int(msg.get("danger", 0))
	var difficulty = null
	for diff in ItemService.difficulties:
		if diff.value == danger:
			difficulty = diff
	if difficulty == null:
		return "unknown danger level: %d" % danger

	var zone := int(msg.get("zone", 0))
	var zone_ids := []
	for zone_data in ZoneService.zones:
		zone_ids.append(zone_data.my_id)
	if not zone in zone_ids:
		return "unknown zone: %d (available: %s)" % [zone, str(zone_ids)]

	# Game-rule overrides. Every reset sets them again, so an option left out of a reset
	# returns to the game default instead of leaking from the previous episode.
	var scaling = msg.get("enemy_scaling")
	if typeof(scaling) != TYPE_DICTIONARY:
		scaling = {}
	ProgressData.settings.enemy_scaling = {
		"health": clamp(float(scaling.get("health", 1.0)), 0.01, 100.0),
		"damage": clamp(float(scaling.get("damage", 1.0)), 0.01, 100.0),
		"speed": clamp(float(scaling.get("speed", 1.0)), 0.01, 100.0),
	}
	ProgressData.settings.keep_lock = bool(msg.get("keep_lock", true))
	var wave_duration = msg.get("wave_duration")
	DebugService.custom_wave_duration = int(wave_duration) if wave_duration != null and int(wave_duration) > 0 else -1

	_restore_unlocks()
	ProgressData.reset_and_save_new_run_state()
	RunData.reset(false)
	RunData.current_zone = zone
	RunData.reset_background()
	RunData.add_character(character, 0)

	if RunData.some_player_has_weapon_slots():
		var choices := _starting_choices(character)
		var wanted := str(msg.get("weapon", ""))
		var choice = choices[0] if not choices.empty() else null
		for candidate in choices:
			if candidate.my_id == wanted:
				choice = candidate
		if wanted != "" and (choice == null or choice.my_id != wanted):
			return "weapon %s is not a starting choice of %s" % [wanted, character_id]
		if choice != null:
			if choice.get_category() == Category.WEAPON:
				var _weapon = RunData.add_weapon(choice, 0, true)
			else:
				RunData.add_item(choice, 0, true)
	RunData.add_starting_items_and_weapons()

	RunData.current_difficulty = difficulty.value
	RunData.reset_elites_spawn()
	RunData.init_elites_spawn()
	RunData.init_events_nightmare()
	RunData.enabled_dlcs = ProgressData.get_active_dlc_ids()
	for effect in difficulty.effects:
		effect.apply(0)
	var uses_ban: bool = bool(msg.get("ban_mode")) if msg.get("ban_mode") != null else RunData.is_ban_mode_active
	for player_data in RunData.players_data:
		player_data.uses_ban = uses_ban
		player_data.remaining_ban_token = int(msg.get("ban_tokens", RunData.BAN_MAX_TOKEN))
	RunData.init_bosses_spawn()
	RunData.current_run_accessibility_settings = ProgressData.settings.enemy_scaling.duplicate()
	ProgressData.load_status = LoadStatus.SAVE_OK
	RunData.is_endless_run = bool(msg.get("endless", false))
	RunData.current_wave = int(clamp(int(msg.get("start_wave", 1)), 1, RunData.nb_of_waves))
	RunData.invulnerable = bool(msg.get("invulnerable", false))
	var starting_materials := int(msg.get("starting_materials", 0))
	if starting_materials > 0:
		RunData.add_gold(starting_materials, 0)
	return ""


func _starting_choices(character) -> Array:
	var choices := []
	choices.append_array(ItemService.get_ordered_starting_weapons(character.starting_weapons))
	choices.append_array(ItemService.get_ordered_starting_items(character.starting_items))
	return choices


func _hello() -> Dictionary:
	var characters := []
	for character in ItemService.characters:
		var starting := []
		for choice in _starting_choices(character):
			starting.append(choice.my_id)
		characters.append({"id": character.my_id, "starting": starting})
	var dangers := []
	for diff in ItemService.difficulties:
		dangers.append(diff.value)
	var zones := []
	for zone_data in ZoneService.zones:
		zones.append(zone_data.my_id)
	# Everything a shop, crate or level-up can offer (mod 0.5.0+), for a fixed id vocabulary.
	var catalog := {"items": [], "weapons": [], "upgrades": []}
	for item in ItemService.items:
		catalog.items.append([item.my_id, item.tier])
	for weapon in ItemService.weapons:
		catalog.weapons.append([weapon.my_id, weapon.tier, "melee" if int(weapon.type) == 0 else "ranged"])
	for upgrade in ItemService.upgrades:
		catalog.upgrades.append([upgrade.my_id, upgrade.tier])
	return {
		"type": "hello",
		"protocol": PROTOCOL_VERSION,
		"mod_version": MOD_VERSION,
		"game_version": ProgressData.VERSION,
		"engine": Engine.get_version_info().string,
		"user_dir": OS.get_user_data_dir(),
		"steam_connected": _options.get("steam_connected", false),
		"pid": OS.get_process_id(),
		"characters": characters,
		"dangers": dangers,
		"zones": zones,
		"catalog": catalog,
	}


# --- Observations -------------------------------------------------------------------

func _build_state(main) -> Dictionary:
	var state := {
		"frame": Engine.get_physics_frames() - _run_start_frame,
		"step": _step,
		"wave": RunData.current_wave,
		"nb_waves": RunData.nb_of_waves,
		"danger": RunData.current_difficulty,
		"counters": {
			"damage_taken": _damage_taken,
			"hits_taken": _hits_taken,
			"dodges": _dodges,
			"healed": _healed,
			"kills": ProgressData.data.enemies_killed - _base_kills,
			"materials_collected": ProgressData.data.materials_collected - _base_materials,
			"trees_killed": ProgressData.data.trees_killed - _base_trees,
		},
	}
	if main == null or main.filename != GAME_SCENE:
		return state

	var zone: Rect2 = ZoneService.get_current_zone_rect()
	state["zone"] = [zone.position.x, zone.position.y, zone.end.x, zone.end.y]
	state["wave_time_left"] = main._wave_timer.time_left
	state["wave_duration"] = main._wave_timer.wait_time
	state["elite_wave"] = main._is_elite_wave
	state["horde_wave"] = main._is_horde_wave
	state["player"] = _player_state(main)

	var spawner = main._entity_spawner
	var enemies := []
	_append_enemies(enemies, spawner.enemies, 0, spawner.charmed_enemies)
	_append_enemies(enemies, spawner.bosses, 1, spawner.charmed_enemies)
	state["enemies"] = enemies
	state["enemy_cols"] = ENEMY_COLS

	var projectiles := []
	if is_instance_valid(main._enemy_projectiles):
		for node in main._enemy_projectiles.get_children():
			_append_projectiles(projectiles, node, 0)
	state["projectiles"] = projectiles

	var births := []
	for birth in main._births_container.get_children():
		if "_current_time_before_spawn" in birth:
			births.append_array([birth.global_position.x, birth.global_position.y, max(0.0, birth._current_time_before_spawn / 60.0), birth.type])
	state["births"] = births

	var materials := []
	for gold in main._active_golds:
		if is_instance_valid(gold) and gold.is_inside_tree():
			materials.append_array([gold.global_position.x, gold.global_position.y, gold.value])
	state["materials"] = materials

	var consumables := []
	for consumable in main._consumables:
		if is_instance_valid(consumable) and consumable.is_inside_tree():
			consumables.append_array([consumable.global_position.x, consumable.global_position.y, _consumable_kind(consumable)])
	state["consumables"] = consumables

	var neutrals := []
	for neutral in spawner.neutrals:
		if is_instance_valid(neutral) and not neutral.dead:
			neutrals.append_array([neutral.global_position.x, neutral.global_position.y, neutral.current_stats.health])
	state["neutrals"] = neutrals

	var structures := []
	for structure in spawner.structures:
		if is_instance_valid(structure):
			structures.append_array([structure.global_position.x, structure.global_position.y])
	state["structures"] = structures

	if not _new_types.empty():
		state["new_types"] = _new_types
		_new_types = {}
	return state


func _player_state(main) -> Dictionary:
	if main._players.empty() or not is_instance_valid(main._players[0]):
		return {"alive": false}
	if _stat_hashes.empty():
		for entry in PLAYER_STATS:
			_stat_hashes.append([entry[0], Keys.get(entry[1])])

	var player = main._players[0]
	var stats := {}
	for entry in _stat_hashes:
		stats[entry[0]] = RunData.get_stat(entry[1], 0)
	return {
		"alive": not player.dead,
		"pos": [player.global_position.x, player.global_position.y],
		"hp": player.current_stats.health,
		"max_hp": player.max_stats.health,
		"move_speed": player.get_move_speed(),
		"iframes": player._invincibility_timer.time_left,
		"level": RunData.get_player_level(0),
		"xp": RunData.get_player_xp(0),
		"xp_next": RunData.get_next_level_xp_needed(0),
		"materials": RunData.get_player_gold(0),
		"bonus_materials": RunData.bonus_gold,
		"weapons": RunData.get_player_weapons(0).size(),
		"stats": stats,
	}


# Row (ENEMY_COLS): x, y, vx, vy, hp, max_hp, type, kind (0 regular, 1 boss, 2 elite),
# attack class (0 contact only, 1 shoots, 2 charges, 3 spawns enemies),
# attack state (0 none, 1 winding up: the "shoot" animation plays and the enemy stands still,
# 2 dashing: a charge is moving it with locked direction).
func _append_enemies(rows: Array, enemies: Array, base_kind: int, charmed: Array) -> void:
	for enemy in enemies:
		if not is_instance_valid(enemy) or enemy.dead:
			continue
		if not charmed.empty() and charmed.has(enemy):
			continue
		var kind := base_kind
		if base_kind == 1 and enemy.get("is_elite"):
			kind = 2
		var velocity: Vector2 = enemy._current_movement.normalized() * enemy.get_move_speed()
		var attack_state := 0
		if enemy._move_locked:
			attack_state = 2
		elif enemy.is_playing_shoot_animation():
			attack_state = 1
		rows.append_array([
			enemy.global_position.x, enemy.global_position.y, velocity.x, velocity.y,
			enemy.current_stats.health, enemy.max_stats.health, _type_index(enemy.enemy_id), kind,
			_attack_class(enemy), attack_state
		])


func _attack_class(enemy) -> int:
	var behavior = enemy.get("_current_attack_behavior")
	if behavior is ChargingAttackBehavior:
		return 2
	if behavior is ShootingAttackBehavior:
		return 1
	if behavior is SpawningAttackBehavior:
		return 3
	return 0


# Row: x, y, vx, vy. Bullet-hell events wrap projectiles in container nodes.
func _append_projectiles(rows: Array, node: Node, depth: int) -> void:
	if "velocity" in node:
		if node.visible:
			rows.append_array([node.global_position.x, node.global_position.y, node.velocity.x, node.velocity.y])
	elif depth < 2:
		for child in node.get_children():
			_append_projectiles(rows, child, depth + 1)


func _consumable_kind(consumable) -> int:
	var data = consumable.consumable_data
	if data == null:
		return 3
	var id: String = data.my_id
	if id.find("legendary") >= 0:
		return 2
	if id.find("item_box") >= 0:
		return 1
	if id.find("fruit") >= 0:
		return 0
	return 3


func _type_index(enemy_id: String) -> int:
	if not _type_ids.has(enemy_id):
		_type_ids[enemy_id] = _type_ids.size()
		_new_types[enemy_id] = _type_ids[enemy_id]
	return _type_ids[enemy_id]


func _hook_player(player) -> void:
	if player.get_instance_id() == _hooked_player_id:
		return
	_hooked_player_id = player.get_instance_id()
	var _e = player.connect("took_damage", self, "_on_player_took_damage")
	_e = player.connect("healed", self, "_on_player_healed")


func _on_player_took_damage(_unit, value, _knockback, _is_crit, is_dodge, is_protected, _armor, _args, _hit_type, _one_shot) -> void:
	if is_dodge:
		_dodges += 1
	elif not is_protected and value > 0:
		_hits_taken += 1
		_damage_taken += int(value)


func _on_player_healed(value, _player_index) -> void:
	_healed += int(value)


# --- Decisions between waves ----------------------------------------------------------

func _describe_item(item) -> Dictionary:
	var effects := []
	for effect in item.effects:
		if effect != null and "key" in effect:
			effects.append([effect.key, effect.value])
	var info := {
		"id": item.my_id,
		"category": "weapon" if item.get_category() == Category.WEAPON else "item",
		"tier": item.tier,
		"value": item.value,
		"effects": effects,
	}
	if item.get_category() == Category.WEAPON:
		var sets := []
		for set_data in item.sets:
			if set_data != null:
				sets.append(set_data.my_id)
		info["weapon_type"] = "melee" if int(item.type) == 0 else "ranged"  # WeaponData.Type: MELEE=0, RANGED=1
		info["sets"] = sets
	return info


func _ban_available() -> bool:
	var player_data = RunData.players_data[0]
	return player_data.uses_ban and player_data.remaining_ban_token > 0 \
		and ChallengeService.is_challenge_completed(ChallengeService.chal_banned_items_hash)


func _ban_state() -> Dictionary:
	var player_data = RunData.players_data[0]
	return {"active": _ban_available(), "tokens": player_data.remaining_ban_token if player_data.uses_ban else 0}


# What the character has between waves (sent with every decision, mod 0.5.0+). HP is full at
# the start of every wave, so max_hp is the number that matters here. Items: one [id, tier] per
# copy owned; weapons: [id, tier, "melee"/"ranged"] in slot order.
func _build_summary() -> Dictionary:
	if _stat_hashes.empty():
		for entry in PLAYER_STATS:
			_stat_hashes.append([entry[0], Keys.get(entry[1])])
	var stats := {}
	for entry in _stat_hashes:
		stats[entry[0]] = RunData.get_stat(entry[1], 0)
	var items := []
	for item in RunData.get_player_items(0):
		items.append([item.my_id, item.tier])
	var weapons := []
	for weapon in RunData.get_player_weapons(0):
		weapons.append([weapon.my_id, weapon.tier, "melee" if int(weapon.type) == 0 else "ranged"])
	return {
		"character": RunData.get_player_character(0).my_id,
		"stats": stats,
		"max_hp": RunData.get_player_max_health(0),
		"level": RunData.get_player_level(0),
		"xp": RunData.get_player_xp(0),
		"xp_next": RunData.get_next_level_xp_needed(0),
		"items": items,
		"weapons": weapons,
		"free_weapon_slots": RunData.get_free_weapon_slots(0),
	}


# Sends a decision and returns the reply, or {} if the reply was reset/quit.
func _ask(decision: Dictionary, key: String) -> Dictionary:
	if key == _last_decision_key:
		_repeated_decisions += 1
	else:
		_last_decision_key = key
		_repeated_decisions = 0
	if _repeated_decisions >= MAX_REPEATED_DECISIONS:
		printerr("[RLBridge] decision %s did not change the game, using the default" % key)
		_repeated_decisions = 0
		return {"type": "choose", "action": "default"}

	decision["type"] = "decision"
	decision["wave"] = RunData.current_wave
	decision["materials"] = RunData.get_player_gold(0)
	decision["build"] = _build_summary()
	_send(decision)
	var reply := _recv()
	if _handle_control(reply):
		return {}
	if reply.get("type") != "choose":
		printerr("[RLBridge] expected choose, got %s" % reply.get("type"))
		return {"type": "choose", "action": "default"}
	return reply


func _tick_upgrades(main) -> void:
	var ui = main._upgrades_ui
	if not is_instance_valid(ui) or not ui.visible or not ui._player_is_choosing[0]:
		return
	var container = ui._player_container1

	if container._upgrades_container.visible:
		var options := []
		var ids := []
		for upgrade in container._old_upgrades:
			options.append(_describe_item(upgrade))
			ids.append(upgrade.my_id)
		var reply := _ask({
			"kind": "level_up",
			"level": container._level,
			"options": options,
			"reroll_price": container._reroll_price,
		}, "level_up:%d:%s" % [container._level, PoolStringArray(ids).join(",")])
		if reply.empty():
			return
		container._button_pressed = false
		if reply.get("action") == "reroll":
			container._on_RerollButton_pressed()
		else:
			var index := int(reply.get("index", 0))
			if index < 0 or index >= container._old_upgrades.size():
				index = 0
			container._on_choose_button_pressed(container._old_upgrades[index])

	elif container._items_container.visible and container._item_data != null:
		var item = container._item_data
		var reply := _ask({
			"kind": "item",
			"item": _describe_item(item),
			"recycle_value": ItemService.get_recycling_value(RunData.current_wave, item.value, 0, item.get_category() == Category.WEAPON),
			"ban": _ban_state(),
		}, "item:%s:%d" % [item.my_id, item.get_instance_id()])
		if reply.empty():
			return
		container._button_pressed = false
		match reply.get("action"):
			"discard":
				container._on_DiscardButton_pressed()
			"ban":
				if _ban_available() and container._ban_button.visible:
					container._on_BanButton_pressed()
				else:
					container._on_DiscardButton_pressed()
			_:
				container._on_TakeButton_pressed()


func _tick_shop(shop) -> void:
	if shop.get_instance_id() != _shop_id:
		_shop_id = shop.get_instance_id()
		_shop_frames = 0
		_shop_actions = 0
	_shop_frames += 1
	if _shop_frames < SHOP_SETTLE_FRAMES or get_tree().paused:
		return
	if _shop_actions >= MAX_SHOP_ACTIONS:
		_shop_go(shop)
		return

	var container = shop._get_shop_items_container(0)
	var items := []
	for i in container._shop_items.size():
		var shop_item = container._shop_items[i]
		if shop_item == null or shop_item.item_data == null:
			continue
		var info := _describe_item(shop_item.item_data)
		info["index"] = i
		info["price"] = shop_item.value
		info["active"] = shop_item.active
		info["locked"] = shop_item.locked
		items.append(info)
	var weapons := []
	var owned := []
	var player_weapons: Array = RunData.get_player_weapons(0)
	for i in player_weapons.size():
		var weapon = player_weapons[i]
		weapons.append(weapon.my_id)
		var detail := _describe_item(weapon)
		detail["index"] = i
		detail["recycle_value"] = ItemService.get_recycling_value(RunData.current_wave, weapon.value, 0, true)
		detail["can_combine"] = RunData.can_combine(weapon, 0)
		owned.append(detail)

	var reply := _ask({
		"kind": "shop",
		"items": items,
		"reroll_price": shop._reroll_price[0],
		"free_weapon_slots": RunData.get_free_weapon_slots(0),
		"weapons": weapons,
		"owned_weapons": owned,
		"locked_count": RunData.get_player_locked_shop_items(0).size(),
		"weapons_locked": RunData.get_player_effect_bool(Keys.lock_current_weapons_hash, 0),
		"ban": _ban_state(),
		"actions_taken": _shop_actions,
	}, "shop:%d:%d:%d" % [_shop_id, _shop_actions, RunData.get_player_gold(0)])
	if reply.empty():
		return
	_shop_actions += 1

	var index := int(reply.get("index", -1))
	match reply.get("action"):
		"buy":
			if index >= 0 and index < container._shop_items.size():
				var shop_item = container._shop_items[index]
				if shop_item != null and shop_item.active:
					container._is_delay_active = false
					shop_item._on_BuyButton_pressed()
		"reroll":
			shop._on_RerollButton_pressed(0)
		"lock", "unlock":
			if index >= 0 and index < container._shop_items.size():
				var shop_item = container._shop_items[index]
				if shop_item != null and shop_item.item_data != null and shop_item.active \
						and shop_item.item_data.is_lockable and shop_item.locked != (reply.action == "lock"):
					shop_item.change_lock_status(reply.action == "lock")
		"ban":
			if index >= 0 and index < container._shop_items.size() and _ban_available():
				var shop_item = container._shop_items[index]
				if shop_item != null and shop_item.active and shop_item.item_data != null \
						and shop_item.item_data.get_category() == Category.ITEM:
					container.on_shop_item_ban_button_pressed(shop_item)
		"recycle", "combine":
			var weapon_index := int(reply.get("weapon_index", -1))
			var current: Array = RunData.get_player_weapons(0)
			if weapon_index >= 0 and weapon_index < current.size():
				var weapon = current[weapon_index]
				if reply.action == "recycle":
					shop._on_item_discard_button_pressed(weapon, 0)
				elif RunData.can_combine(weapon, 0):
					shop._on_item_combine_button_pressed(weapon, 0)
		_:
			_shop_go(shop)


func _shop_go(shop) -> void:
	_stale_scene_id = shop.get_instance_id()
	shop._on_GoButton_pressed(0)
