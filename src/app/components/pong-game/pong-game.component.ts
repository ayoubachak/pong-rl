import { Component, ElementRef, ViewChild, OnDestroy, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GameSettings } from '../game-menu/game-menu.component';

interface Ball {
  x: number;
  y: number;
  dx: number;
  dy: number;
  radius: number;
  speed: number;
}

interface Paddle {
  x: number;
  y: number;
  width: number;
  height: number;
  speed: number;
}

interface GameState {
  player1Score: number;
  player2Score: number;
  isGameRunning: boolean;
  isPaused: boolean;
  gameMode: 'vs-bot' | 'vs-player';
  difficulty: 'easy' | 'medium' | 'hard';
  winner: string | null;
}

@Component({
  selector: 'app-pong-game',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './pong-game.component.html',
  styleUrls: ['./pong-game.component.css']
})
export class PongGameComponent implements OnInit, OnDestroy {
  @ViewChild('gameCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @Input() settings!: GameSettings;
  @Output() gameEnded = new EventEmitter<void>();

  private ctx!: CanvasRenderingContext2D;
  private animationId!: number;
  private readonly keysPressed: Set<string> = new Set();

  canvas = {
    width: 800,
    height: 400
  };

  ball: Ball = {
    x: 400,
    y: 200,
    dx: 5,
    dy: 3,
    radius: 8,
    speed: 5
  };

  paddle1: Paddle = {
    x: 20,
    y: 175,
    width: 10,
    height: 80,
    speed: 8
  };

  paddle2: Paddle = {
    x: 770,
    y: 175,
    width: 10,
    height: 80,
    speed: 8
  };

  gameState: GameState = {
    player1Score: 0,
    player2Score: 0,
    isGameRunning: false,
    isPaused: false,
    gameMode: 'vs-bot',
    difficulty: 'medium',
    winner: null
  };

  private botDifficulty = 0.8; // AI reaction speed
  private botReactionDelay = 0;

  ngOnInit() {
    this.initializeGame();
    this.setupEventListeners();
    this.startGame();
  }

  ngOnDestroy() {
    this.cleanup();
  }

  private initializeGame() {
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.canvas.width;
    canvas.height = this.canvas.height;
    this.ctx = canvas.getContext('2d')!;

    // Apply settings
    if (this.settings) {
      this.gameState.gameMode = this.settings.gameMode;
      this.gameState.difficulty = this.settings.difficulty;
      this.ball.speed = this.settings.ballSpeed;
      this.paddle1.speed = this.settings.paddleSpeed;
      this.paddle2.speed = this.settings.paddleSpeed;
      
      // Adjust bot difficulty
      switch (this.settings.difficulty) {
        case 'easy':
          this.botDifficulty = 0.6;
          break;
        case 'medium':
          this.botDifficulty = 0.8;
          break;
        case 'hard':
          this.botDifficulty = 0.95;
          break;
      }
    }

    this.resetBall();
  }

  private setupEventListeners() {
    document.addEventListener('keydown', this.onKeyDown.bind(this));
    document.addEventListener('keyup', this.onKeyUp.bind(this));
  }

  private onKeyDown(event: KeyboardEvent) {
    this.keysPressed.add(event.key);
    
    // Pause/Resume game
    if (event.key === ' ') {
      event.preventDefault();
      this.togglePause();
    }
    
    // Restart game
    if (event.key === 'r' || event.key === 'R') {
      event.preventDefault();
      this.restartGame();
    }
  }

  private onKeyUp(event: KeyboardEvent) {
    this.keysPressed.delete(event.key);
  }

  private startGame() {
    this.gameState.isGameRunning = true;
    this.gameLoop();
  }

  private gameLoop() {
    if (!this.gameState.isGameRunning) return;

    this.update();
    this.render();
    this.animationId = requestAnimationFrame(() => this.gameLoop());
  }

  private update() {
    if (this.gameState.isPaused || this.gameState.winner) return;

    this.updatePaddles();
    this.updateBall();
    this.checkCollisions();
    this.checkScore();
  }

  private updatePaddles() {
    // Player 1 controls (W/S or Arrow Up/Down)
    if (this.keysPressed.has('w') || this.keysPressed.has('W') || this.keysPressed.has('ArrowUp')) {
      this.paddle1.y = Math.max(0, this.paddle1.y - this.paddle1.speed);
    }
    if (this.keysPressed.has('s') || this.keysPressed.has('S') || this.keysPressed.has('ArrowDown')) {
      this.paddle1.y = Math.min(this.canvas.height - this.paddle1.height, this.paddle1.y + this.paddle1.speed);
    }

    if (this.gameState.gameMode === 'vs-player') {
      // Player 2 controls (Arrow keys only for player 2)
      if (this.keysPressed.has('ArrowUp')) {
        this.paddle2.y = Math.max(0, this.paddle2.y - this.paddle2.speed);
      }
      if (this.keysPressed.has('ArrowDown')) {
        this.paddle2.y = Math.min(this.canvas.height - this.paddle2.height, this.paddle2.y + this.paddle2.speed);
      }
    } else {
      // Bot AI
      this.updateBot();
    }
  }

  private updateBot() {
    const ballCenterY = this.ball.y;
    const paddleCenterY = this.paddle2.y + this.paddle2.height / 2;
    const distance = ballCenterY - paddleCenterY;

    // Add some reaction delay for realism
    this.botReactionDelay++;
    if (this.botReactionDelay < 3) return;
    this.botReactionDelay = 0;

    // Bot only reacts when ball is moving towards it
    if (this.ball.dx > 0) {
      const moveSpeed = this.paddle2.speed * this.botDifficulty;
      
      if (Math.abs(distance) > 10) {
        if (distance > 0) {
          this.paddle2.y = Math.min(this.canvas.height - this.paddle2.height, this.paddle2.y + moveSpeed);
        } else {
          this.paddle2.y = Math.max(0, this.paddle2.y - moveSpeed);
        }
      }
    }

    // Progressive difficulty - bot gets better as game progresses
    const totalScore = this.gameState.player1Score + this.gameState.player2Score;
    if (totalScore > 0) {
      this.botDifficulty = Math.min(0.98, this.botDifficulty + 0.01);
    }
  }

  private updateBall() {
    this.ball.x += this.ball.dx;
    this.ball.y += this.ball.dy;

    // Ball collision with top and bottom walls
    if (this.ball.y - this.ball.radius <= 0 || this.ball.y + this.ball.radius >= this.canvas.height) {
      this.ball.dy = -this.ball.dy;
    }
  }

  private checkCollisions() {
    // Ball collision with paddles
    if (this.ballCollidesWithPaddle(this.paddle1) || this.ballCollidesWithPaddle(this.paddle2)) {
      this.ball.dx = -this.ball.dx;
      
      // Add some angle variation based on where the ball hits the paddle
      const paddle = this.ball.x < this.canvas.width / 2 ? this.paddle1 : this.paddle2;
      const relativeIntersectY = (paddle.y + paddle.height / 2) - this.ball.y;
      const normalizedIntersectY = relativeIntersectY / (paddle.height / 2);
      this.ball.dy = -normalizedIntersectY * this.ball.speed * 0.5;
      
      // Increase ball speed slightly after each hit
      this.ball.dx *= 1.02;
      this.ball.dy *= 1.02;
    }
  }

  private ballCollidesWithPaddle(paddle: Paddle): boolean {
    return this.ball.x - this.ball.radius <= paddle.x + paddle.width &&
           this.ball.x + this.ball.radius >= paddle.x &&
           this.ball.y - this.ball.radius <= paddle.y + paddle.height &&
           this.ball.y + this.ball.radius >= paddle.y;
  }

  private checkScore() {
    if (this.ball.x < 0) {
      this.gameState.player2Score++;
      this.resetBall();
    } else if (this.ball.x > this.canvas.width) {
      this.gameState.player1Score++;
      this.resetBall();
    }

    // Check for winner
    if (this.gameState.player1Score >= this.settings.winScore) {
      this.gameState.winner = this.gameState.gameMode === 'vs-bot' ? 'Player' : 'Player 1';
      this.endGame();
    } else if (this.gameState.player2Score >= this.settings.winScore) {
      this.gameState.winner = this.gameState.gameMode === 'vs-bot' ? 'Bot' : 'Player 2';
      this.endGame();
    }
  }

  private resetBall() {
    this.ball.x = this.canvas.width / 2;
    this.ball.y = this.canvas.height / 2;
    this.ball.dx = (Math.random() > 0.5 ? 1 : -1) * this.ball.speed;
    this.ball.dy = (Math.random() - 0.5) * this.ball.speed;
  }

  private render() {
    // Clear canvas
    this.ctx.fillStyle = '#0f0f23';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw center line
    this.ctx.setLineDash([10, 10]);
    this.ctx.strokeStyle = '#00ff00';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvas.width / 2, 0);
    this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // Draw paddles
    this.ctx.fillStyle = '#00ff00';
    this.ctx.fillRect(this.paddle1.x, this.paddle1.y, this.paddle1.width, this.paddle1.height);
    this.ctx.fillRect(this.paddle2.x, this.paddle2.y, this.paddle2.width, this.paddle2.height);

    // Draw ball
    this.ctx.beginPath();
    this.ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
    this.ctx.fill();

    // Draw scores
    this.ctx.font = '48px Courier New';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(this.gameState.player1Score.toString(), this.canvas.width / 4, 60);
    this.ctx.fillText(this.gameState.player2Score.toString(), (3 * this.canvas.width) / 4, 60);

    // Draw game info
    if (this.gameState.isPaused) {
      this.ctx.font = '24px Courier New';
      this.ctx.fillText('PAUSED', this.canvas.width / 2, this.canvas.height / 2);
      this.ctx.font = '16px Courier New';
      this.ctx.fillText('Press SPACE to resume', this.canvas.width / 2, this.canvas.height / 2 + 30);
    }

    if (this.gameState.winner) {
      this.ctx.font = '32px Courier New';
      this.ctx.fillText(`${this.gameState.winner} Wins!`, this.canvas.width / 2, this.canvas.height / 2);
      this.ctx.font = '16px Courier New';
      this.ctx.fillText('Press R to restart or ESC to return to menu', this.canvas.width / 2, this.canvas.height / 2 + 40);
    }
  }

  togglePause() {
    if (!this.gameState.winner) {
      this.gameState.isPaused = !this.gameState.isPaused;
    }
  }

  restartGame() {
    this.gameState.player1Score = 0;
    this.gameState.player2Score = 0;
    this.gameState.isPaused = false;
    this.gameState.winner = null;
    this.resetBall();
    this.paddle1.y = 175;
    this.paddle2.y = 175;
    
    // Reset bot difficulty
    switch (this.settings.difficulty) {
      case 'easy':
        this.botDifficulty = 0.6;
        break;
      case 'medium':
        this.botDifficulty = 0.8;
        break;
      case 'hard':
        this.botDifficulty = 0.95;
        break;
    }
  }

  exitToMenu() {
    this.gameEnded.emit();
  }

  private endGame() {
    // Game ends but keeps running for display
  }

  private cleanup() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    document.removeEventListener('keydown', this.onKeyDown.bind(this));
    document.removeEventListener('keyup', this.onKeyUp.bind(this));
  }
}