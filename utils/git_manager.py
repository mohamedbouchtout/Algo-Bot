"""
Class that manages git operations for the project, such as pulling updates, checking status, 
and handling branches. This class can be used to automate git tasks within the project workflow.
"""

import logging
import os
import subprocess
from datetime import datetime
from core.connection import ConnectionManager

# Setup logging
logger = logging.getLogger()

class GitManager:
    def __init__(self, ib, connection_manager: ConnectionManager, config, params):
        self.ib = ib
        self.connection_manager = connection_manager
        self.config = config
        self.params = params
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(self, last_git_check: datetime) -> datetime:
        """Commit and push changes to git if interval has passed"""
        # Only commit once per hour
        # if (datetime.now() - last_git_check).total_seconds() > self.config['git']['commit_interval']:
        #     self.git_commit_and_push("Auto-commit: Trading bot update")
        #     last_git_check = datetime.now()
        
        # Check for updates only once per day and after market close
        if datetime.now().hour == 16 and datetime.now().minute < 20:  # After market close
            if self.check_for_updates():
                self.logger.info("Updating and restarting bot to apply new changes...")
                if self.pull_updates():
                    self.connection_manager.restart_bot()

        return last_git_check

    def check_for_updates(self) -> bool:
        """Check if remote has new commits"""
        try:
            # Fetch latest from remote
            subprocess.run(['git', 'fetch', 'origin'], cwd=self.base_dir, check=True, capture_output=True)
            
            # Compare local and remote
            result = subprocess.run(
                ['git', 'rev-list', f'HEAD...origin/{self.config["git"]["main_branch"]}', '--count'],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits_behind = int(result.stdout.strip())
            
            if commits_behind > 0:
                logger.info(f"{commits_behind} new commit(s) available")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not check for updates: {e}")
            return False
    
    def pull_updates(self) -> bool:
        """Pull latest changes from git"""
        try:
            # Fetch latest
            logger.info("Pulling latest changes...")

            subprocess.run(['git', 'fetch', 'origin'], cwd=self.base_dir, check=True, capture_output=True)
            result = subprocess.run(
                ['git', 'merge', f'origin/{self.config["git"]["main_branch"]}', '-m', f'Auto-merge from {self.config["git"]["main_branch"]}'],
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("Merge successful!")
                
                # Push the merge commit
                subprocess.run(
                    ['git', 'push'],
                    cwd=self.base_dir,
                    check=True,
                    capture_output=True
                )
                
                return True
            
            else:
                # Merge conflict detected
                logger.error(f"Merge conflict:\n{result.stderr}")
                
                # Try to auto-resolve
                if self.auto_resolve_conflicts():
                    logger.info("Conflicts auto-resolved")
                    return True
                else:
                    logger.error("Manual intervention required")
                    return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to merge updates: {e}")
            
            # Abort merge if it's in progress
            subprocess.run(
                ['git', 'merge', '--abort'],
                cwd=self.base_dir,
                check=False
            )
            
            return False

    def auto_resolve_conflicts(self) -> bool:
        """
        Attempt to auto-resolve merge conflicts
        Strategy: Accept main for code files, keep ours for data files
        """
        try:
            # Check which files have conflicts
            status = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            conflict_files = []
            for line in status.stdout.split('\n'):
                if line.startswith('UU '):  # Unmerged files
                    conflict_files.append(line[3:].strip())
            
            if not conflict_files:
                return True
            
            logger.info(f"Resolving conflicts in: {conflict_files}")
            
            for file in conflict_files:
                # Strategy: Code files → use main, Data files → use ours
                if file.endswith('.py') or file.endswith('.json') and 'config/' in file:
                    # Code/config files: Accept changes from main
                    logger.info(f"  {file}: Using version from {self.config['git']['main_branch']}")
                    subprocess.run(
                        ['git', 'checkout', '--theirs', file],
                        cwd=self.base_dir,
                        check=True
                    )
                elif 'data/' in file or file.endswith('.log'):
                    # Data/log files: Keep our version
                    logger.info(f"  {file}: Keeping local version")
                    subprocess.run(
                        ['git', 'checkout', '--ours', file],
                        cwd=self.base_dir,
                        check=True
                    )
                else:
                    # Unknown file: Use main version
                    logger.warning(f"  {file}: Unknown type, using {self.config['git']['main_branch']} version")
                    subprocess.run(
                        ['git', 'checkout', '--theirs', file],
                        cwd=self.base_dir,
                        check=True
                    )
            
            # Complete the merge
            subprocess.run(
                ['git', 'add', '.'],
                cwd=self.base_dir,
                check=True
            )
            subprocess.run(
                ['git', 'commit', '-m', 'Auto-resolved merge conflicts'],
                cwd=self.base_dir,
                check=True
            )
            subprocess.run(
                ['git', 'push'],
                cwd=self.base_dir,
                check=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-resolve conflicts: {e}")
            
            # Abort the merge
            subprocess.run(
                ['git', 'merge', '--abort'],
                cwd=self.base_dir,
                check=False
            )
            
            return False

    def git_commit_and_push(self, message=None):
        """Commit and push changes to git"""
        try:
            if message is None:
                message = f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Check if there are changes to commit
            status = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            if not status.stdout.strip():
                logger.debug("No changes to commit")
                return True

            # Commit with message
            subprocess.run(['git', 'commit', '-m', message], cwd=self.base_dir, check=True, capture_output=True)
            
            # Push to remote
            subprocess.run(['git', 'push'], cwd=self.base_dir, check=True)
            
            logger.info(f"Successfully pushed: {message}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git error: {e}")
            return False