import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

type GameMode = 'vs-bot' | 'vs-player';
type Difficulty = 'easy' | 'medium' | 'hard';
type CurrentView = 'main' | 'settings';

export interface GameSettings {
  gameMode: GameMode;
  difficulty: Difficulty;
  ballSpeed: number;
  paddleSpeed: number;
  winScore: number;
}

@Component({
  selector: 'app-game-menu',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './game-menu.component.html',
  styleUrls: ['./game-menu.component.css']
})
export class GameMenuComponent {
  @Output() startGame = new EventEmitter<GameSettings>();
  @Output() watchAILearn = new EventEmitter<void>();

  currentView: CurrentView = 'main';
  
  gameSettings: GameSettings = {
    gameMode: 'vs-bot',
    difficulty: 'medium',
    ballSpeed: 5,
    paddleSpeed: 8,
    winScore: 5
  };

  // Legacy settings property for existing template
  settings = {
    difficulty: 'medium' as Difficulty,
    winScore: 5
  };

  startVsBot() {
    this.gameSettings.gameMode = 'vs-bot';
    this.gameSettings.difficulty = this.settings.difficulty;
    this.gameSettings.winScore = this.settings.winScore;
    this.startGame.emit(this.gameSettings);
  }

  onWatchAILearn() {
    this.watchAILearn.emit();
  }

  openSettings() {
    this.currentView = 'settings';
  }

  backToMain() {
    this.currentView = 'main';
  }

  updateDifficulty(difficulty: Difficulty) {
    this.settings.difficulty = difficulty;
    this.gameSettings.difficulty = difficulty;
  }

  onStartGame() {
    this.startGame.emit(this.gameSettings);
  }

  toggleSettings() {
    this.currentView = this.currentView === 'main' ? 'settings' : 'main';
  }
}
