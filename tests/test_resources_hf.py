"""
Tests for HuggingFace-based model zoo functions.

These tests are written in TDD style - they test the new HuggingFace API
functions before implementation. All tests use mocking since the functions
don't exist yet.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestListModels:
    """Tests for list_models() function."""

    @patch('grelu.resources.HfApi')
    def test_list_models_returns_model_repos(self, mock_hf_api):
        """Test that list_models returns a list of model repository IDs."""
        from grelu.resources import list_models

        # Setup mock collection with items
        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = [
            Mock(item_id="Genentech/human-atac-catlas-model"),
            Mock(item_id="Genentech/borzoi-model"),
            Mock(item_id="Genentech/enformer-model"),
        ]
        mock_api.get_collection.return_value = mock_collection

        result = list_models()

        assert isinstance(result, list)
        assert "Genentech/human-atac-catlas-model" in result
        assert "Genentech/borzoi-model" in result
        assert "Genentech/enformer-model" in result

    @patch('grelu.resources.HfApi')
    def test_list_models_empty_collection(self, mock_hf_api):
        """Test that list_models handles empty collection gracefully."""
        from grelu.resources import list_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = []
        mock_api.get_collection.return_value = mock_collection

        result = list_models()

        assert isinstance(result, list)
        assert len(result) == 0

    @patch('grelu.resources.HfApi')
    def test_list_models_calls_correct_collection(self, mock_hf_api):
        """Test that list_models calls get_collection with correct parameters."""
        from grelu.resources import list_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = []
        mock_api.get_collection.return_value = mock_collection

        list_models()

        # Verify the API was called
        mock_api.get_collection.assert_called_once()


class TestListDatasets:
    """Tests for list_datasets() function."""

    @patch('grelu.resources.HfApi')
    def test_list_datasets_returns_dataset_repos(self, mock_hf_api):
        """Test that list_datasets returns a list of dataset repository IDs."""
        from grelu.resources import list_datasets

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = [
            Mock(item_id="Genentech/human-atac-catlas-data"),
            Mock(item_id="Genentech/borzoi-data"),
        ]
        mock_api.get_collection.return_value = mock_collection

        result = list_datasets()

        assert isinstance(result, list)
        assert "Genentech/human-atac-catlas-data" in result
        assert "Genentech/borzoi-data" in result

    @patch('grelu.resources.HfApi')
    def test_list_datasets_empty_collection(self, mock_hf_api):
        """Test that list_datasets handles empty collection gracefully."""
        from grelu.resources import list_datasets

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = []
        mock_api.get_collection.return_value = mock_collection

        result = list_datasets()

        assert isinstance(result, list)
        assert len(result) == 0

    @patch('grelu.resources.HfApi')
    def test_list_datasets_calls_correct_collection(self, mock_hf_api):
        """Test that list_datasets calls get_collection with correct parameters."""
        from grelu.resources import list_datasets

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_collection = Mock()
        mock_collection.items = []
        mock_api.get_collection.return_value = mock_collection

        list_datasets()

        # Verify the API was called
        mock_api.get_collection.assert_called_once()


class TestDownloadModel:
    """Tests for download_model() function."""

    @patch('grelu.resources.hf_hub_download')
    def test_download_model_returns_path(self, mock_download):
        """Test that download_model returns the local file path."""
        from grelu.resources import download_model

        mock_download.return_value = "/path/to/model.ckpt"

        result = download_model(repo_id="Genentech/test-model")

        assert result == "/path/to/model.ckpt"

    @patch('grelu.resources.hf_hub_download')
    def test_download_model_with_custom_filename(self, mock_download):
        """Test that download_model respects custom filename."""
        from grelu.resources import download_model

        mock_download.return_value = "/path/to/custom_model.ckpt"

        result = download_model(repo_id="Genentech/test-model", filename="custom_model.ckpt")

        assert result == "/path/to/custom_model.ckpt"
        mock_download.assert_called_once()
        # Check that filename was passed correctly
        call_kwargs = mock_download.call_args
        assert "custom_model.ckpt" in str(call_kwargs)

    @patch('grelu.resources.hf_hub_download')
    def test_download_model_default_filename(self, mock_download):
        """Test that download_model uses default filename when not specified."""
        from grelu.resources import download_model

        mock_download.return_value = "/path/to/model.ckpt"

        download_model(repo_id="Genentech/test-model")

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        # Default should be model.ckpt
        assert "model.ckpt" in str(call_kwargs)


class TestDownloadDataset:
    """Tests for download_dataset() function."""

    @patch('grelu.resources.hf_hub_download')
    def test_download_dataset_returns_path(self, mock_download):
        """Test that download_dataset returns the local file path."""
        from grelu.resources import download_dataset

        mock_download.return_value = "/path/to/data.h5ad"

        result = download_dataset(repo_id="Genentech/test-data")

        assert result == "/path/to/data.h5ad"

    @patch('grelu.resources.hf_hub_download')
    def test_download_dataset_with_custom_filename(self, mock_download):
        """Test that download_dataset respects custom filename."""
        from grelu.resources import download_dataset

        mock_download.return_value = "/path/to/custom_data.csv"

        result = download_dataset(repo_id="Genentech/test-data", filename="custom_data.csv")

        assert result == "/path/to/custom_data.csv"
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert "custom_data.csv" in str(call_kwargs)

    @patch('grelu.resources.hf_hub_download')
    def test_download_dataset_calls_hf_hub_download_correctly(self, mock_download):
        """Test that download_dataset passes correct repo_id to hf_hub_download."""
        from grelu.resources import download_dataset

        mock_download.return_value = "/path/to/data.h5ad"

        download_dataset(repo_id="Genentech/human-atac-catlas-data")

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert "Genentech/human-atac-catlas-data" in str(call_kwargs)


class TestLoadModel:
    """Tests for load_model() function."""

    @patch('grelu.resources.LightningModel')
    @patch('grelu.resources.hf_hub_download')
    def test_load_model_downloads_and_loads(self, mock_download, mock_lightning):
        """Test that load_model downloads checkpoint and loads the model."""
        from grelu.resources import load_model

        mock_download.return_value = "/path/to/model.ckpt"
        mock_model = Mock()
        mock_lightning.load_from_checkpoint.return_value = mock_model

        result = load_model(repo_id="Genentech/test-model")

        assert result == mock_model
        mock_download.assert_called_once()
        mock_lightning.load_from_checkpoint.assert_called_once()

    @patch('grelu.resources.LightningModel')
    @patch('grelu.resources.hf_hub_download')
    def test_load_model_with_device(self, mock_download, mock_lightning):
        """Test that load_model passes device parameter correctly."""
        from grelu.resources import load_model

        mock_download.return_value = "/path/to/model.ckpt"
        mock_model = Mock()
        mock_lightning.load_from_checkpoint.return_value = mock_model

        load_model(repo_id="Genentech/test-model", device="cuda:0")

        call_kwargs = mock_lightning.load_from_checkpoint.call_args
        assert "cuda:0" in str(call_kwargs) or call_kwargs.kwargs.get('map_location') == "cuda:0"

    @patch('grelu.resources.LightningModel')
    @patch('grelu.resources.hf_hub_download')
    def test_load_model_default_device_is_cpu(self, mock_download, mock_lightning):
        """Test that load_model uses CPU as default device."""
        from grelu.resources import load_model

        mock_download.return_value = "/path/to/model.ckpt"
        mock_model = Mock()
        mock_lightning.load_from_checkpoint.return_value = mock_model

        load_model(repo_id="Genentech/test-model")

        call_kwargs = mock_lightning.load_from_checkpoint.call_args
        # Default device should be 'cpu'
        assert "cpu" in str(call_kwargs) or call_kwargs.kwargs.get('map_location') == "cpu"

    @patch('grelu.resources.LightningModel')
    @patch('grelu.resources.hf_hub_download')
    def test_load_model_with_custom_filename(self, mock_download, mock_lightning):
        """Test that load_model respects custom checkpoint filename."""
        from grelu.resources import load_model

        mock_download.return_value = "/path/to/custom.ckpt"
        mock_model = Mock()
        mock_lightning.load_from_checkpoint.return_value = mock_model

        load_model(repo_id="Genentech/test-model", filename="custom.ckpt")

        call_kwargs = mock_download.call_args
        assert "custom.ckpt" in str(call_kwargs)


class TestGetDatasetsByModel:
    """Tests for get_datasets_by_model() function."""

    @patch('grelu.resources.HfApi')
    def test_get_datasets_by_model_parses_metadata(self, mock_hf_api):
        """Test that get_datasets_by_model parses model card metadata correctly."""
        from grelu.resources import get_datasets_by_model

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.datasets = ["Genentech/human-atac-catlas-data"]
        mock_api.model_info.return_value = mock_info

        result = get_datasets_by_model(repo_id="Genentech/human-atac-catlas-model")

        assert result == ["Genentech/human-atac-catlas-data"]

    @patch('grelu.resources.HfApi')
    def test_get_datasets_by_model_multiple_datasets(self, mock_hf_api):
        """Test that get_datasets_by_model handles multiple linked datasets."""
        from grelu.resources import get_datasets_by_model

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.datasets = [
            "Genentech/dataset-1",
            "Genentech/dataset-2",
            "Genentech/dataset-3",
        ]
        mock_api.model_info.return_value = mock_info

        result = get_datasets_by_model(repo_id="Genentech/test-model")

        assert len(result) == 3
        assert "Genentech/dataset-1" in result
        assert "Genentech/dataset-2" in result
        assert "Genentech/dataset-3" in result

    @patch('grelu.resources.HfApi')
    def test_get_datasets_by_model_no_datasets(self, mock_hf_api):
        """Test that get_datasets_by_model handles models with no linked datasets."""
        from grelu.resources import get_datasets_by_model

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.datasets = []
        mock_api.model_info.return_value = mock_info

        result = get_datasets_by_model(repo_id="Genentech/test-model")

        assert result == []

    @patch('grelu.resources.HfApi')
    def test_get_datasets_by_model_no_card_data(self, mock_hf_api):
        """Test that get_datasets_by_model handles missing card_data gracefully."""
        from grelu.resources import get_datasets_by_model

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = None
        mock_api.model_info.return_value = mock_info

        result = get_datasets_by_model(repo_id="Genentech/test-model")

        assert result == []


class TestGetBaseModels:
    """Tests for get_base_models() function."""

    @patch('grelu.resources.HfApi')
    def test_get_base_models_parses_metadata(self, mock_hf_api):
        """Test that get_base_models parses model card metadata correctly."""
        from grelu.resources import get_base_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.base_model = ["Genentech/borzoi-model"]
        mock_api.model_info.return_value = mock_info

        result = get_base_models(repo_id="Genentech/finetuned-model")

        assert result == ["Genentech/borzoi-model"]

    @patch('grelu.resources.HfApi')
    def test_get_base_models_single_base_model(self, mock_hf_api):
        """Test that get_base_models handles single base model string."""
        from grelu.resources import get_base_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.base_model = "Genentech/enformer-model"
        mock_api.model_info.return_value = mock_info

        result = get_base_models(repo_id="Genentech/finetuned-model")

        # Should return as a list even if input is a string
        assert isinstance(result, list)
        assert "Genentech/enformer-model" in result

    @patch('grelu.resources.HfApi')
    def test_get_base_models_no_base_model(self, mock_hf_api):
        """Test that get_base_models handles models with no base model."""
        from grelu.resources import get_base_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = Mock()
        mock_info.card_data.base_model = None
        mock_api.model_info.return_value = mock_info

        result = get_base_models(repo_id="Genentech/test-model")

        assert result == []

    @patch('grelu.resources.HfApi')
    def test_get_base_models_no_card_data(self, mock_hf_api):
        """Test that get_base_models handles missing card_data gracefully."""
        from grelu.resources import get_base_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_info = Mock()
        mock_info.card_data = None
        mock_api.model_info.return_value = mock_info

        result = get_base_models(repo_id="Genentech/test-model")

        assert result == []


class TestUtilityFunctions:
    """Tests for utility functions that should still work after migration."""

    def test_get_blacklist_file_hg38(self):
        """Test that get_blacklist_file returns correct path for hg38."""
        from grelu.resources import get_blacklist_file

        result = get_blacklist_file("hg38")

        assert "hg38" in result
        assert "blacklist" in result.lower()
        assert result.endswith(".bed")

    def test_get_blacklist_file_hg19(self):
        """Test that get_blacklist_file returns correct path for hg19."""
        from grelu.resources import get_blacklist_file

        result = get_blacklist_file("hg19")

        assert "hg19" in result
        assert "blacklist" in result.lower()
        assert result.endswith(".bed")

    def test_get_blacklist_file_mm10(self):
        """Test that get_blacklist_file returns correct path for mm10."""
        from grelu.resources import get_blacklist_file

        result = get_blacklist_file("mm10")

        assert "mm10" in result
        assert "blacklist" in result.lower()
        assert result.endswith(".bed")

    def test_get_meme_file_path_hocomoco_v12(self):
        """Test that get_meme_file_path returns correct path for hocomoco_v12."""
        from grelu.resources import get_meme_file_path

        result = get_meme_file_path("hocomoco_v12")

        assert result.endswith(".meme")
        assert "H12CORE" in result

    def test_get_meme_file_path_hocomoco_v13(self):
        """Test that get_meme_file_path returns correct path for hocomoco_v13."""
        from grelu.resources import get_meme_file_path

        result = get_meme_file_path("hocomoco_v13")

        assert result.endswith(".meme")
        assert "H13CORE" in result

    def test_get_meme_file_path_consensus(self):
        """Test that get_meme_file_path returns correct path for consensus."""
        from grelu.resources import get_meme_file_path

        result = get_meme_file_path("consensus")

        assert result.endswith(".meme")
        assert "consensus" in result.lower()


class TestErrorHandling:
    """Tests for error handling in HuggingFace functions."""

    @patch('grelu.resources.HfApi')
    def test_list_models_handles_api_error(self, mock_hf_api):
        """Test that list_models handles API errors gracefully."""
        from grelu.resources import list_models

        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        mock_api.get_collection.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            list_models()

    @patch('grelu.resources.hf_hub_download')
    def test_download_model_handles_download_error(self, mock_download):
        """Test that download_model handles download errors."""
        from grelu.resources import download_model

        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception):
            download_model(repo_id="Genentech/nonexistent-model")

    @patch('grelu.resources.hf_hub_download')
    def test_download_dataset_handles_download_error(self, mock_download):
        """Test that download_dataset handles download errors."""
        from grelu.resources import download_dataset

        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception):
            download_dataset(repo_id="Genentech/nonexistent-data")

    @patch('grelu.resources.LightningModel')
    @patch('grelu.resources.hf_hub_download')
    def test_load_model_handles_checkpoint_error(self, mock_download, mock_lightning):
        """Test that load_model handles checkpoint loading errors."""
        from grelu.resources import load_model

        mock_download.return_value = "/path/to/model.ckpt"
        mock_lightning.load_from_checkpoint.side_effect = Exception("Invalid checkpoint")

        with pytest.raises(Exception):
            load_model(repo_id="Genentech/corrupted-model")
