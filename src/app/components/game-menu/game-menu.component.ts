import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface GameSettings {
  gameMode: 'vs-bot' | 'vs-player';
  difficulty: 'easy' | 'medium' | 'hard';
  ballSpeed: number;
  paddleSpeed: number;
  winScore: number;
}

@Component({
  selector: 'app-game-menu',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './game-menu.component.html',
  styleUrls: ['./game-menu.component.css']
})
export class GameMenuComponent {
  @Output() startGame = new EventEmitter<GameSettings>();
  @Output() showSettings = new EventEmitter<void>();

  currentView: 'main' | 'settings' = 'main';
  
  settings: GameSettings = {
    gameMode: 'vs-bot',
    difficulty: 'medium',
    ballSpeed: 5,
    paddleSpeed: 8,
    winScore: 5
  };

  startVsBot() {
    this.settings.gameMode = 'vs-bot';
    this.startGame.emit(this.settings);
  }

  startVsPlayer() {
    this.settings.gameMode = 'vs-player';
    this.startGame.emit(this.settings);
  }

  openSettings() {
    this.currentView = 'settings';
  }

  backToMain() {
    this.currentView = 'main';
  }

  updateDifficulty(difficulty: 'easy' | 'medium' | 'hard') {
    this.settings.difficulty = difficulty;
    // Adjust speeds based on difficulty
    switch (difficulty) {
      case 'easy':
        this.settings.ballSpeed = 3;
        this.settings.paddleSpeed = 10;
        break;
      case 'medium':
        this.settings.ballSpeed = 5;
        this.settings.paddleSpeed = 8;
        break;
      case 'hard':
        this.settings.ballSpeed = 7;
        this.settings.paddleSpeed = 6;
        break;
    }
  }
}
