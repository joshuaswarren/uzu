//! Linux credential store: a secret saved through Settings must read back, and clear() must remove it.
#![cfg(target_os = "linux")]

use uzu::settings::{SettingKind, Settings};

#[test]
fn secret_roundtrip_uses_a_real_store() {
    let settings = Settings::new("com.trymirai.uzu.linux-test".to_string()).expect("settings");
    settings.save(SettingKind::Secret, "api_key".to_string(), Some("s3cret".to_string())).expect("save");
    assert_eq!(settings.load(SettingKind::Secret, "api_key".to_string()).expect("load"), Some("s3cret".to_string()));
    // a second Settings instance sees the same value, which the in-memory mock store would not provide
    let again = Settings::new("com.trymirai.uzu.linux-test".to_string()).expect("settings");
    assert_eq!(again.load(SettingKind::Secret, "api_key".to_string()).expect("load"), Some("s3cret".to_string()));
    again.save(SettingKind::Secret, "api_key".to_string(), None).expect("remove");
}
