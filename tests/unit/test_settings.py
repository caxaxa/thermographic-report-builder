"""Tests for Settings configuration — env vars, backward compat, bucket suffixes."""
import os
import pytest


class TestGetBucketSuffix:
    def test_dev_environment(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "dev")
        # Re-import to pick up the new env value
        from thermographic_report_builder.config.settings import get_bucket_suffix

        assert get_bucket_suffix() == "dev"

    def test_prod_environment(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "prod")
        from thermographic_report_builder.config.settings import get_bucket_suffix

        assert get_bucket_suffix() == "prod"

    def test_default_is_prod(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        from thermographic_report_builder.config.settings import get_bucket_suffix

        assert get_bucket_suffix() == "prod"

    def test_staging_maps_to_prod(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "staging")
        from thermographic_report_builder.config.settings import get_bucket_suffix

        assert get_bucket_suffix() == "prod"


class TestSettingsRequiredFields:
    def test_project_id_required(self, monkeypatch):
        """Settings should fail without project_id."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_PROJECT_ID", raising=False)
        monkeypatch.delenv("PROJECT_ID", raising=False)
        with pytest.raises(Exception):
            Settings(user_id="test-user")

    def test_user_id_required(self, monkeypatch):
        """Settings should fail without user_id."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_USER_ID", raising=False)
        monkeypatch.delenv("ORG_ID", raising=False)
        monkeypatch.delenv("USER_ID", raising=False)
        with pytest.raises(Exception):
            Settings(project_id="test-project")


class TestSettingsBackwardCompat:
    def test_old_project_id_env(self, monkeypatch, tmp_path):
        """PROJECT_ID should be accepted when SOLAR_PROJECT_ID is not set."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_PROJECT_ID", raising=False)
        monkeypatch.setenv("PROJECT_ID", "old-style-id")
        monkeypatch.setenv("SOLAR_USER_ID", "test-user")
        s = Settings(work_dir=tmp_path / "work")
        assert s.project_id == "old-style-id"

    def test_old_org_id_env(self, monkeypatch, tmp_path):
        """ORG_ID should be accepted when SOLAR_USER_ID is not set."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_USER_ID", raising=False)
        monkeypatch.setenv("ORG_ID", "old-org")
        monkeypatch.setenv("SOLAR_PROJECT_ID", "test-project")
        s = Settings(work_dir=tmp_path / "work")
        assert s.user_id == "old-org"

    def test_old_user_id_env(self, monkeypatch, tmp_path):
        """USER_ID should be accepted when SOLAR_USER_ID and ORG_ID are not set."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_USER_ID", raising=False)
        monkeypatch.delenv("ORG_ID", raising=False)
        monkeypatch.setenv("USER_ID", "old-user")
        monkeypatch.setenv("SOLAR_PROJECT_ID", "test-project")
        s = Settings(work_dir=tmp_path / "work")
        assert s.user_id == "old-user"

    def test_old_orthos_bucket_env(self, monkeypatch, tmp_path):
        """ORTHOS_BUCKET should be accepted when SOLAR_ORTHOS_BUCKET is not set."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_ORTHOS_BUCKET", raising=False)
        monkeypatch.setenv("ORTHOS_BUCKET", "my-custom-bucket")
        monkeypatch.setenv("SOLAR_PROJECT_ID", "test-project")
        monkeypatch.setenv("SOLAR_USER_ID", "test-user")
        s = Settings(work_dir=tmp_path / "work")
        assert s.orthos_bucket == "my-custom-bucket"

    def test_old_aws_region_env(self, monkeypatch, tmp_path):
        """AWS_DEFAULT_REGION should be accepted when SOLAR_AWS_REGION is not set."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.delenv("SOLAR_AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        monkeypatch.setenv("SOLAR_PROJECT_ID", "test-project")
        monkeypatch.setenv("SOLAR_USER_ID", "test-user")
        s = Settings(work_dir=tmp_path / "work")
        assert s.aws_region == "eu-west-1"

    def test_new_env_takes_precedence(self, monkeypatch, tmp_path):
        """SOLAR_* vars should override old-style vars."""
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.setenv("SOLAR_PROJECT_ID", "new-style")
        monkeypatch.setenv("PROJECT_ID", "old-style")
        monkeypatch.setenv("SOLAR_USER_ID", "test-user")
        s = Settings(work_dir=tmp_path / "work")
        assert s.project_id == "new-style"


class TestSettingsDefaults:
    @pytest.fixture
    def settings(self, monkeypatch, tmp_path):
        from thermographic_report_builder.config.settings import Settings

        monkeypatch.setenv("SOLAR_PROJECT_ID", "test-project")
        monkeypatch.setenv("SOLAR_USER_ID", "test-user")
        return Settings(work_dir=tmp_path / "work")

    def test_default_aws_region(self, settings):
        assert settings.aws_region == "us-east-2"

    def test_default_report_language(self, settings):
        assert settings.report_language == "pt-BR"

    def test_default_company_name(self, settings):
        assert settings.company_name == "Aisol"

    def test_default_downscale_factor(self, settings):
        assert settings.orthophoto_downscale_factor == 0.25

    def test_work_dir_created(self, settings):
        assert settings.work_dir.exists()

    def test_compile_only_default_false(self, settings):
        assert settings.compile_only is False

    def test_enable_mesh_fallback_default_false(self, settings):
        assert settings.enable_mesh_fallback is False
