// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::encryption::key_management::kms::{
    KmsClient, KmsClientFactory, KmsClientRef, KmsConnectionConfig,
};
use crate::errors::ParquetError;
use crate::errors::Result;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use ring::aead::{Aad, LessSafeKey, UnboundKey, AES_128_GCM, NONCE_LEN};
use ring::rand::{SecureRandom, SystemRandom};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// KMS client implementation for unit tests, which is compatible
/// with the C++ Arrow LocalWrapKmsClient
pub struct TestKmsClient {
    key_map: HashMap<String, Vec<u8>>,
    keys_wrapped: Arc<Mutex<usize>>,
    keys_unwrapped: Arc<Mutex<usize>>,
}

pub struct TestKmsClientFactory {
    key_map: HashMap<String, Vec<u8>>,
    invocations: Mutex<Vec<String>>,
    keys_wrapped: Arc<Mutex<usize>>,
    keys_unwrapped: Arc<Mutex<usize>>,
}

impl TestKmsClientFactory {
    pub fn with_default_keys() -> Self {
        let mut key_map = HashMap::default();
        key_map.insert("kf".to_owned(), "0123456789012345".as_bytes().to_vec());
        key_map.insert("kc1".to_owned(), "1234567890123450".as_bytes().to_vec());
        key_map.insert("kc2".to_owned(), "1234567890123451".as_bytes().to_vec());

        Self {
            key_map,
            invocations: Mutex::new(Vec::new()),
            keys_wrapped: Arc::new(Mutex::new(0)),
            keys_unwrapped: Arc::new(Mutex::new(0)),
        }
    }

    /// Get the access keys used to create clients.
    /// Provided for unit testing
    pub fn invocations(&self) -> Vec<String> {
        self.invocations.lock().unwrap().clone()
    }

    /// Get the number of times a key was wrapped with a KMS client created by this factory
    pub fn keys_wrapped(&self) -> usize {
        self.keys_wrapped.lock().unwrap().clone()
    }

    /// Get the number of times a key was unwrapped with a KMS client created by this factory
    pub fn keys_unwrapped(&self) -> usize {
        self.keys_unwrapped.lock().unwrap().clone()
    }
}

impl KmsClientFactory for TestKmsClientFactory {
    fn create_client(&self, kms_connection_config: &KmsConnectionConfig) -> Result<KmsClientRef> {
        {
            let mut invocations = self.invocations.lock().unwrap();
            invocations.push(kms_connection_config.key_access_token().to_owned());
        }
        Ok(Arc::new(TestKmsClient::new(
            self.key_map.clone(),
            self.keys_wrapped.clone(),
            self.keys_unwrapped.clone(),
        )))
    }
}

impl TestKmsClient {
    pub fn new(
        key_map: HashMap<String, Vec<u8>>,
        keys_wrapped: Arc<Mutex<usize>>,
        keys_unwrapped: Arc<Mutex<usize>>,
    ) -> Self {
        Self {
            key_map,
            keys_wrapped,
            keys_unwrapped,
        }
    }

    fn get_key(&self, master_key_identifier: &str) -> Result<LessSafeKey> {
        let key = self
            .key_map
            .get(master_key_identifier)
            .ok_or_else(|| general_err!("Invalid master key '{}'", master_key_identifier))?;
        let key = UnboundKey::new(&AES_128_GCM, &key)
            .map_err(|e| general_err!("Error creating AES key '{}'", e))?;
        Ok(LessSafeKey::new(key))
    }
}

impl KmsClient for TestKmsClient {
    fn wrap_key(&self, key_bytes: &[u8], master_key_identifier: &str) -> Result<String> {
        let key = self.get_key(master_key_identifier)?;
        let aad = master_key_identifier.as_bytes();
        let rng = SystemRandom::new();

        let mut nonce = [0u8; NONCE_LEN];
        rng.fill(&mut nonce)?;
        let nonce = ring::aead::Nonce::assume_unique_for_key(nonce);

        let tag_len = key.algorithm().tag_len();
        let mut ciphertext = Vec::with_capacity(NONCE_LEN + key_bytes.len() + tag_len);
        ciphertext.extend_from_slice(nonce.as_ref());
        ciphertext.extend_from_slice(&key_bytes);
        let tag =
            key.seal_in_place_separate_tag(nonce, Aad::from(aad), &mut ciphertext[NONCE_LEN..])?;
        ciphertext.extend_from_slice(tag.as_ref());
        let encoded = BASE64_STANDARD.encode(&ciphertext);

        {
            let mut guard = self.keys_wrapped.lock().unwrap();
            *guard += 1;
        }

        Ok(encoded)
    }

    fn unwrap_key(&self, wrapped_key: &str, master_key_identifier: &str) -> Result<Vec<u8>> {
        let key = self.get_key(master_key_identifier)?;
        let aad = master_key_identifier.as_bytes();

        let wrapped_key = BASE64_STANDARD
            .decode(wrapped_key)
            .map_err(|e| general_err!("Error base64 decoding wrapped key: {}", e))?;
        let nonce = ring::aead::Nonce::try_assume_unique_for_key(&wrapped_key[..NONCE_LEN])?;

        let mut plaintext = Vec::with_capacity(wrapped_key.len() - NONCE_LEN);
        plaintext.extend_from_slice(&wrapped_key[NONCE_LEN..]);

        let tag_len = key.algorithm().tag_len();
        key.open_in_place(nonce, Aad::from(aad), &mut plaintext)?;
        plaintext.resize(plaintext.len() - tag_len, 0u8);

        {
            let mut guard = self.keys_unwrapped.lock().unwrap();
            *guard += 1;
        }

        Ok(plaintext)
    }
}
