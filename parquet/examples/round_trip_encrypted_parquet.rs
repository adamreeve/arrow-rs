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

use arrow::array::{Float32Builder, StructArray, UInt64Builder};
use arrow::datatypes::DataType;
use arrow::datatypes::{Field, Schema};
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ArrowWriter as ParquetWriter;
use parquet::encryption::key_management::crypto_factory::{
    CryptoFactory, DecryptionConfiguration, EncryptionConfigurationBuilder,
};
use parquet::encryption::key_management::kms::{KmsClient, KmsConnectionConfig};
use parquet::errors::{ParquetError, Result};
use parquet::file::properties::WriterProperties;
use ring::aead::{Aad, LessSafeKey, UnboundKey, AES_128_GCM, NONCE_LEN};
use ring::rand::{SecureRandom, SystemRandom};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

fn main() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("encrypted_example.parquet");

    // Create a CryptoFactory that will use our Demo KMS client
    let crypto_factory = CryptoFactory::new(DemoKmsClient::create);

    // Specify any options required to connect to our KMS.
    // These are ignored by the `DemoKmsClient` but shown here for illustration.
    // The KMS instance ID and URL will be stored in the Parquet encryption metadata
    // so don't need to be specified if you are only reading files.
    let connection_config = Arc::new(
        KmsConnectionConfig::builder()
            .set_kms_instance_id("kms1".into())
            .set_kms_instance_url("https://example.com/kms".into())
            .set_key_access_token("secret_token".into())
            .set_custom_kms_conf_option("custom_option".into(), "some_value".into())
            .build(),
    );

    // Create an encryption configuration that will encrypt the footer with the "kf" key,
    // the "x" column with the "kc1" key, and the "y" column with the "kc2" key,
    // while leaving the "id" column unencrypted.
    let encryption_config = EncryptionConfigurationBuilder::new("kf".into())
        .add_column_key("kc1".into(), vec!["x".into()])
        .build();

    // Use the CryptoFactory to generate file encryption properties
    let encryption_properties =
        crypto_factory.file_encryption_properties(connection_config.clone(), &encryption_config)?;
    let write_properties = WriterProperties::builder()
        .with_file_encryption_properties(encryption_properties)
        .build();

    write_file(&file_path, write_properties)?;

    // Use the CryptoFactory to generate file decryption properties
    let decryption_config = DecryptionConfiguration::default();
    let decryption_properties =
        crypto_factory.file_decryption_properties(connection_config, decryption_config)?;
    let read_options =
        ArrowReaderOptions::new().with_file_decryption_properties(decryption_properties);

    read_file(&file_path, read_options)?;

    Ok(())
}

/// Write a Parquet file to the specified path, using the provided properties
fn write_file(path: &PathBuf, properties: WriterProperties) -> Result<()> {
    let file = File::create(path)?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]));

    let mut writer = ParquetWriter::try_new(file, schema.clone(), Some(properties))?;

    let mut id_builder = UInt64Builder::new();
    let mut x_builder = Float32Builder::new();
    let mut y_builder = Float32Builder::new();
    let num_rows = 10;
    for i in 0..num_rows {
        id_builder.append_value(i);
        x_builder.append_value(i as f32 / 10.0);
        y_builder.append_value(i as f32 / 100.0);
    }
    writer.write(
        &StructArray::new(
            schema.fields().clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(x_builder.finish()),
                Arc::new(y_builder.finish()),
            ],
            None,
        )
        .into(),
    )?;
    writer.flush()?;
    writer.close()?;

    Ok(())
}

/// Read a Parquet file at the specified path, using the provided options
fn read_file(path: &PathBuf, options: ArrowReaderOptions) -> Result<()> {
    let file = File::open(path)?;

    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options)?;
    let record_reader = builder.build()?;
    for batch in record_reader {
        let batch = batch?;
        println!("Read batch: {:?}", batch);
    }
    Ok(())
}

/// Example KMS client that uses in-memory AES keys.
/// A real KMS client should interact with a Key Management Server to encrypt and decrypt keys.
pub struct DemoKmsClient {
    key_map: HashMap<String, Vec<u8>>,
}

impl DemoKmsClient {
    pub fn create(_config: &KmsConnectionConfig) -> Result<Arc<dyn KmsClient>> {
        let mut key_map = HashMap::default();
        key_map.insert("kf".to_owned(), "0123456789012345".as_bytes().to_vec());
        key_map.insert("kc1".to_owned(), "1234567890123450".as_bytes().to_vec());
        key_map.insert("kc2".to_owned(), "1234567890123451".as_bytes().to_vec());

        Ok(Arc::new(Self { key_map }))
    }

    /// Get the AES key corresponding to a key identifier
    fn get_key(&self, master_key_identifier: &str) -> Result<LessSafeKey> {
        let key = self.key_map.get(master_key_identifier).ok_or_else(|| {
            ParquetError::General(format!("Invalid master key '{}'", master_key_identifier))
        })?;
        let key = UnboundKey::new(&AES_128_GCM, key)
            .map_err(|e| ParquetError::General(format!("Error creating AES key '{}'", e)))?;
        Ok(LessSafeKey::new(key))
    }
}

impl KmsClient for DemoKmsClient {
    /// Take a randomly generated key and encrypt it using the specified master key
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
        ciphertext.extend_from_slice(key_bytes);
        let tag =
            key.seal_in_place_separate_tag(nonce, Aad::from(aad), &mut ciphertext[NONCE_LEN..])?;
        ciphertext.extend_from_slice(tag.as_ref());
        let encoded = BASE64_STANDARD.encode(&ciphertext);

        Ok(encoded)
    }

    /// Take an encrypted key and decrypt it using the specified master key identifier
    fn unwrap_key(&self, wrapped_key: &str, master_key_identifier: &str) -> Result<Vec<u8>> {
        let key = self.get_key(master_key_identifier)?;
        let aad = master_key_identifier.as_bytes();

        let wrapped_key = BASE64_STANDARD.decode(wrapped_key).map_err(|e| {
            ParquetError::General(format!("Error base64 decoding wrapped key: {}", e))
        })?;
        let nonce = ring::aead::Nonce::try_assume_unique_for_key(&wrapped_key[..NONCE_LEN])?;

        let mut plaintext = Vec::with_capacity(wrapped_key.len() - NONCE_LEN);
        plaintext.extend_from_slice(&wrapped_key[NONCE_LEN..]);

        let tag_len = key.algorithm().tag_len();
        key.open_in_place(nonce, Aad::from(aad), &mut plaintext)?;
        plaintext.resize(plaintext.len() - tag_len, 0u8);

        Ok(plaintext)
    }
}
