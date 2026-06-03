from src.pipelines.attack_pipeline import select_malicious_clients, is_attack_active, get_attack_config
from src.configs.config import CONFIG


# -- Test 1: select_malicious_clients count
print("Test 1: select_malicious_clients count")
ids = select_malicious_clients(50, 0.30, seed=42)
assert len(ids) == 15, f"Expected 15, got {len(ids)}"
print(f"  Selected {len(ids)} clients at 30% ratio  [OK]")


# -- Test 2: deterministic with same seed
print("\nTest 2: Deterministic selection")
ids2 = select_malicious_clients(50, 0.30, seed=42)
assert ids == ids2
print("  Same seed gives same result  [OK]")


# -- Test 3: different seed gives different result
print("\nTest 3: Different seed gives different result")
ids3 = select_malicious_clients(50, 0.30, seed=99)
assert ids != ids3
print("  Different seeds give different clients  [OK]")


# -- Test 4: 0% attacker ratio returns empty list
print("\nTest 4: 0% attacker ratio")
ids_empty = select_malicious_clients(50, 0.0, seed=42)
assert len(ids_empty) == 0
print("  Empty list for 0% ratio  [OK]")


# -- Test 5: is_attack_active boundary
print("\nTest 5: is_attack_active boundaries")
start = CONFIG["attack"]["attack_start_round"]
assert not is_attack_active(start - 1, start)
assert is_attack_active(start, start)
assert is_attack_active(start + 5, start)
print(f"  Correct at round {start-1} (False), {start} (True), {start+5} (True)  [OK]")


# -- Test 6: get_attack_config for benign client
print("\nTest 6: get_attack_config — benign client")
malicious = select_malicious_clients(50, 0.30, seed=42)
benign_id = next(i for i in range(50) if i not in malicious)
cfg = get_attack_config(benign_id, malicious, server_round=15)
assert cfg["is_poisoned"] is False
print(f"  Client {benign_id} (benign): is_poisoned={cfg['is_poisoned']}  [OK]")


# -- Test 7: get_attack_config for malicious client before attack start
print("\nTest 7: get_attack_config — malicious but before attack_start_round")
start = CONFIG["attack"]["attack_start_round"]
cfg2 = get_attack_config(malicious[0], malicious, server_round=start - 1)
assert cfg2["is_poisoned"] is False
print(f"  Client {malicious[0]} at round {start-1}: is_poisoned={cfg2['is_poisoned']}  [OK]")


# -- Test 8: get_attack_config for active malicious client
print("\nTest 8: get_attack_config — active attack")
cfg3 = get_attack_config(malicious[0], malicious, server_round=start)
assert cfg3["is_poisoned"] is True
assert "attack_type" in cfg3
assert "source_class" in cfg3
print(f"  Client {malicious[0]} at round {start}: is_poisoned={cfg3['is_poisoned']}, type={cfg3['attack_type']}  [OK]")


print("\nAll attack_pipeline tests passed.")
